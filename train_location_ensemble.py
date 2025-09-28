#!/usr/bin/env python3
"""
Anchor-wise, class-agnostic learnable fusion for bounding boxes.

Train:
  - Head 1 (coords): regress GT xywh (normalized) from anchor-centered, cross-model features
    *fit only on anchors whose nearest GT IoU >= label_iou_pos*
  - Head 2 (score): regress IoU(anchor, nearest GT) in [0,1] for all anchors

Infer (no clustering):
  - Use every model detection as an anchor
  - Build same features
  - Predict refined xywh and score (IoU regressed)
  - Apply class-agnostic NMS; write COCO predictions and (optionally) evaluate

Additions:
  - Per-model evaluation (train + test) with simplified metrics:
      Precision & Recall at IoU = 0.1, 0.3, 0.5, 0.7

Assumptions:
  - All prediction JSONs are COCO-style predictions (list of dicts) or a full dict with "annotations"
  - Their image_ids are aligned to GT; if not, we try to remap via filename if "images" present

Author: you :)
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---------------------------
# Geometry helpers
# ---------------------------

def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0

def nms_xywh(dets: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """
    dets: [N,4] xywh
    scores: [N]
    returns keep indices
    """
    if len(dets) == 0:
        return []
    boxes = np.stack([xywh_to_xyxy(b) for b in dets], axis=0)  # [N,4]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        i_box = boxes[i]
        rest = boxes[order[1:]]
        xx1 = np.maximum(i_box[0], rest[:, 0])
        yy1 = np.maximum(i_box[1], rest[:, 1])
        xx2 = np.minimum(i_box[2], rest[:, 2])
        yy2 = np.minimum(i_box[3], rest[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (i_box[2] - i_box[0]) * (i_box[3] - i_box[1])
        area_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        ovr = inter / (area_i + area_r - inter + 1e-9)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep


# ---------------------------
# IO / mapping
# ---------------------------

def load_coco_gt(gt_json: str):
    coco = COCO(gt_json)
    images_by_id = {img_id: coco.imgs[img_id] for img_id in coco.imgs}
    gts_by_image = defaultdict(list)
    for ann in coco.anns.values():
        gts_by_image[ann["image_id"]].append(ann["bbox"])  # xywh
    return coco, images_by_id, gts_by_image

def _read_predictions_json(pred_path: str) -> Tuple[List[dict], Any]:
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"], data
    raise ValueError(f"Unsupported predictions format: {pred_path}")

def _maybe_remap_ids_by_filename(
    dets: List[dict], pred_dict: Any, images_by_id_gt: Dict[int, dict]
) -> List[dict]:
    gt_ids = set(images_by_id_gt.keys())
    if all(d.get("image_id") in gt_ids for d in dets):
        return dets
    if not (isinstance(pred_dict, dict) and "images" in pred_dict):
        return dets
    # pred image_id -> filename
    pid2fname = {img["id"]: img["file_name"] for img in pred_dict.get("images", [])}
    # filename -> gt id
    fname2gtid = {v["file_name"]: k for k, v in images_by_id_gt.items()}
    fixed = []
    for d in dets:
        iid = d.get("image_id")
        if iid in gt_ids:
            fixed.append(d); continue
        fname = pid2fname.get(iid)
        if fname is not None and fname in fname2gtid:
            nd = dict(d)
            nd["image_id"] = fname2gtid[fname]
            fixed.append(nd)
        else:
            # drop if cannot map
            pass
    return fixed

def load_predictions_per_model(pred_paths: List[str], images_by_id_gt: Dict[int, dict]):
    """
    Returns:
      models (List[str]), per_model (Dict[model -> Dict[image_id -> List[(xywh, score)]]])
    """
    per_model = {}
    models = []
    for p in pred_paths:
        model = os.path.splitext(os.path.basename(p))[0]
        dets, raw = _read_predictions_json(p)
        dets = _maybe_remap_ids_by_filename(dets, raw, images_by_id_gt)
        imgmap = defaultdict(list)
        for d in dets:
            iid = d.get("image_id")
            if iid not in images_by_id_gt:
                continue
            bbox = d.get("bbox", [0,0,0,0])
            score = float(d.get("score", 1.0))
            imgmap[iid].append((bbox, score))
        models.append(model)
        per_model[model] = imgmap
    return models, per_model


# ---------------------------
# Simplified metrics (Precision/Recall at IoU 0.1/0.3/0.5/0.7)
# ---------------------------

def compute_pr_at_thresholds(coco_gt: COCO, dets: List[dict], thresholds=[0.1, 0.3, 0.5, 0.7]):
    results = {}
    if not dets:
        return {t: {"precision": 0.0, "recall": 0.0} for t in thresholds}
    coco_dt = coco_gt.loadRes(dets)
    for thr in thresholds:
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.useCats = 0
        ev.params.iouThrs = np.array([thr])
        ev.evaluate(); ev.accumulate()
        # precision shape: [T,R,K,A,M] where T=1 here
        prec = ev.eval.get("precision")
        rec  = ev.eval.get("recall")
        if prec is None or rec is None:
            results[thr] = {"precision": 0.0, "recall": 0.0}
            continue
        prec_vals = prec[prec > -1]
        rec_vals  = rec[rec > -1]
        prec_mean = float(np.mean(prec_vals)) if prec_vals.size else 0.0
        rec_mean  = float(np.mean(rec_vals))  if rec_vals.size  else 0.0
        results[thr] = {"precision": prec_mean, "recall": rec_mean}
    return results

def print_metrics(name: str, metrics: Dict[float, Dict[str, float]]):
    print(f"\n[{name}] simplified metrics (class-agnostic):")
    for thr in sorted(metrics.keys()):
        vals = metrics[thr]
        print(f" IoU={thr:.1f} -> Precision={vals['precision']:.3f}, Recall={vals['recall']:.3f}")


def eval_single_models(coco_gt: COCO, pred_paths: List[str], images_by_id: Dict[int, dict], split="train"):
    print(f"\n=== Single-model evaluations on {split} (Precision/Recall at IoU 0.1/0.3/0.5/0.7) ===")
    for p in pred_paths:
        model_name = os.path.splitext(os.path.basename(p))[0]
        dets, raw = _read_predictions_json(p)
        dets = _maybe_remap_ids_by_filename(dets, raw, images_by_id)
        metrics = compute_pr_at_thresholds(coco_gt, dets)
        print_metrics(model_name, metrics)


# ---------------------------
# Feature building
# ---------------------------

def best_match_to_anchor(anchor_xyxy, cand_xywh_list: List[Tuple[List[float], float]]):
    """
    Among candidate boxes, find the one with maximum IoU with the anchor (xyxy).
    Returns (xywh, score, iou_to_anchor) or (None, 0.0, 0.0) if empty.
    """
    if not cand_xywh_list:
        return None, 0.0, 0.0
    best_iou = -1.0
    best = None
    for (b, s) in cand_xywh_list:
        iou = iou_xyxy(anchor_xyxy, xywh_to_xyxy(b))
        if iou > best_iou:
            best_iou = iou
            best = (b, float(s), float(iou))
    return best if best is not None else (None, 0.0, 0.0)

def nearest_gt(anchor_xyxy, gt_xywh_list: List[List[float]]) -> Tuple[int, float]:
    """
    Returns index of nearest GT by IoU and the IoU value. If none, (-1, 0.0)
    """
    if not gt_xywh_list:
        return -1, 0.0
    best_iou = -1.0
    best_idx = -1
    for i, g in enumerate(gt_xywh_list):
        iou = iou_xyxy(anchor_xyxy, xywh_to_xyxy(g))
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx, float(max(0.0, best_iou))

def build_feature_vector(
    anchor_xywh, anchor_score, anchor_model_idx, model_names,
    per_model_boxes_for_image: Dict[str, List[Tuple[List[float], float]]],
    img_w: int, img_h: int,
    min_neighbor_iou: float = 0.05
) -> np.ndarray:
    """
    Create a deterministic feature vector for an anchor by fetching best-overlap
    boxes from every model relative to the anchor.
    Per-model features: [x/W, y/H, w/W, h/H, score, iou_to_anchor, present_flag]
    Global: [anchor_x/W, anchor_y/H, anchor_w/W, anchor_h/H, anchor_score] + one-hot anchor model
    """
    W, H = float(img_w), float(img_h)
    ax, ay, aw, ah = anchor_xywh
    feat = []

    # per-model features
    for m in model_names:
        cand = per_model_boxes_for_image.get(m, [])
        b, s, iou = best_match_to_anchor(xywh_to_xyxy(anchor_xywh), cand)
        if b is None or iou < min_neighbor_iou:
            feat.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # present_flag=0
        else:
            bx, by, bw, bh = b
            feat.extend([
                bx / max(W,1e-6), by / max(H,1e-6),
                bw / max(W,1e-6), bh / max(H,1e-6),
                s, iou, 1.0  # present_flag=1
            ])

    # global anchor geometry + score
    feat.extend([ax / max(W,1e-6), ay / max(H,1e-6), aw / max(W,1e-6), ah / max(H,1e-6), float(anchor_score)])

    # anchor model one-hot
    onehot = [0.0] * len(model_names)
    if 0 <= anchor_model_idx < len(model_names):
        onehot[anchor_model_idx] = 1.0
    feat.extend(onehot)

    return np.array(feat, dtype=np.float32)


# ---------------------------
# Dataset builder (anchor-wise)
# ---------------------------

def build_training_sets(
    images_by_id: Dict[int, dict],
    gts_by_image: Dict[int, List[List[float]]],
    models: List[str],
    per_model: Dict[str, Dict[int, List[Tuple[List[float], float]]]],
    label_iou_pos: float = 0.4,
    min_neighbor_iou: float = 0.05,
    anchor_score_thresh: float = 0.0,
):
    """
    Build training data for:
      - coord_reg (X-> y_box in normalized xywh) using only anchors whose nearest GT IoU >= label_iou_pos
      - iou_reg   (X-> y_iou in [0,1]) on all anchors
    """
    X_coords, y_coords = [], []
    X_iou, y_iou = [], []

    for iid, meta in images_by_id.items():
        W, H = meta["width"], meta["height"]
        gt_list = gts_by_image.get(iid, [])
        # collect anchors = union of all model detections over this image
        for mi, m in enumerate(models):
            for (b, s) in per_model.get(m, {}).get(iid, []):
                if s < anchor_score_thresh:
                    continue
                feat = build_feature_vector(
                    anchor_xywh=b, anchor_score=s, anchor_model_idx=mi,
                    model_names=models,
                    per_model_boxes_for_image={mm: per_model.get(mm, {}).get(iid, []) for mm in models},
                    img_w=W, img_h=H,
                    min_neighbor_iou=min_neighbor_iou
                )
                # nearest GT to anchor
                gi, iou = nearest_gt(xywh_to_xyxy(b), gt_list)
                # IoU regression target (always defined)
                X_iou.append(feat); y_iou.append(iou)
                # Coord regression target (only when anchor has a reasonably close GT)
                if gi >= 0 and iou >= label_iou_pos:
                    gx, gy, gw, gh = gt_list[gi]
                    ybox = np.array([gx / W, gy / H, gw / W, gh / H], dtype=np.float32)
                    X_coords.append(feat); y_coords.append(ybox)

    X_coords = np.stack(X_coords, axis=0) if len(X_coords) else None
    y_coords = np.stack(y_coords, axis=0) if len(y_coords) else None
    X_iou = np.stack(X_iou, axis=0) if len(X_iou) else None
    y_iou = np.array(y_iou, dtype=np.float32) if len(y_iou) else None

    print(f"[train] coord samples: {0 if X_coords is None else len(X_coords)} (anchors with IoU >= {label_iou_pos})")
    print(f"[train] iou   samples: {0 if X_iou    is None else len(X_iou)} (all anchors)")

    return X_coords, y_coords, X_iou, y_iou


# ---------------------------
# Models
# ---------------------------

def make_coord_regressor(method: str = "xgb"):
    if method == "xgb" and HAVE_XGB:
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=-1, objective="reg:squarederror"
            )
        )
    # fallback: RF
    return MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        )
    )

def make_iou_regressor(method: str = "xgb"):
    if method == "xgb" and HAVE_XGB:
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=-1, objective="reg:squarederror"
            ))
        ])
    # fallback: RF
    return RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42
    )


# ---------------------------
# Inference on a split (no clustering)
# ---------------------------

def infer_split(
    images_by_id: Dict[int, dict],
    models: List[str],
    per_model: Dict[str, Dict[int, List[Tuple[List[float], float]]]],
    coord_reg,
    iou_reg,
    min_neighbor_iou: float = 0.05,
    anchor_score_thresh: float = 0.0,
    nms_iou: float = 0.5,
) -> List[dict]:
    outputs = []

    for iid, meta in images_by_id.items():
        W, H = meta["width"], meta["height"]
        feats = []
        # anchors_xywh kept for clarity (not used directly)
        anchors_xywh = []
        # build per-anchor features
        for mi, m in enumerate(models):
            dets = per_model.get(m, {}).get(iid, [])
            for (b, s) in dets:
                if s < anchor_score_thresh:
                    continue
                feat = build_feature_vector(
                    anchor_xywh=b, anchor_score=s, anchor_model_idx=mi,
                    model_names=models,
                    per_model_boxes_for_image={mm: per_model.get(mm, {}).get(iid, []) for mm in models},
                    img_w=W, img_h=H,
                    min_neighbor_iou=min_neighbor_iou
                )
                feats.append(feat)
                anchors_xywh.append(b)

        if not feats:
            continue

        X = np.stack(feats, axis=0)

        # predict coords (normalized) and score (IoU-like)
        ybox = coord_reg.predict(X)           # [N,4] normalized
        yiou = iou_reg.predict(X).reshape(-1) # [N]

        # denormalize boxes
        boxes = []
        for row in ybox:
            x = max(0.0, min(1.0, float(row[0]))) * W
            y = max(0.0, min(1.0, float(row[1]))) * H
            w = max(0.0, min(1.0, float(row[2]))) * W
            h = max(0.0, min(1.0, float(row[3]))) * H
            boxes.append([x, y, w, h])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.clip(yiou, 0.0, 1.0)

        # NMS (class-agnostic)
        keep = nms_xywh(boxes, scores, iou_thr=nms_iou)
        for k in keep:
            outputs.append({
                "image_id": iid,
                "category_id": 1,          # ignored in class-agnostic eval
                "bbox": [float(v) for v in boxes[k].tolist()],
                "score": float(scores[k]),
            })

    return outputs


# ---------------------------
# COCO eval (class-agnostic, full summarize)
# ---------------------------

def coco_eval_classagnostic(coco_gt: COCO, preds: List[dict]):
    dt = coco_gt.loadRes(preds if len(preds) > 0 else [])
    ev = COCOeval(coco_gt, dt, "bbox")
    ev.params.useCats = 0  # class-agnostic
    ev.evaluate(); ev.accumulate(); ev.summarize()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser("Anchor-wise, class-agnostic learnable box fusion")
    ap.add_argument("--train_gt", required=True, help="COCO GT json (train)")
    ap.add_argument("--train_preds", required=True, nargs="+", help="N model prediction jsons (train)")
    ap.add_argument("--test_gt", required=True, help="COCO GT json (test)")
    ap.add_argument("--test_preds", required=True, nargs="+", help="N model prediction jsons (test)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--label_iou_pos", type=float, default=0.4, help="Anchor->GT IoU to include sample for coordinate regression")
    ap.add_argument("--min_neighbor_iou", type=float, default=0.05, help="Min IoU for a model's box to contribute features to an anchor")
    ap.add_argument("--anchor_score_thresh", type=float, default=0.0, help="Drop raw anchors with score below this")
    ap.add_argument("--nms_iou", type=float, default=0.5, help="NMS IoU at inference")
    ap.add_argument("--coord_reg", choices=["xgb","rf"], default="xgb", help="Regressor for coordinates")
    ap.add_argument("--iou_reg", choices=["xgb","rf"], default="xgb", help="Regressor for score/IoU")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- TRAIN ----
    coco_tr, images_tr, gts_tr = load_coco_gt(args.train_gt)
    models_tr, per_model_tr = load_predictions_per_model(args.train_preds, images_tr)
    print("[train] models:", models_tr)

    # Per-model baselines on TRAIN (simplified metrics)
    eval_single_models(coco_tr, args.train_preds, images_tr, split="train")

    # Build training data + fit regressors
    Xc, yc, Xi, yi = build_training_sets(
        images_by_id=images_tr,
        gts_by_image=gts_tr,
        models=models_tr,
        per_model=per_model_tr,
        label_iou_pos=args.label_iou_pos,
        min_neighbor_iou=args.min_neighbor_iou,
        anchor_score_thresh=args.anchor_score_thresh,
    )

    if Xc is None or yc is None or Xi is None or yi is None:
        raise RuntimeError("Insufficient training data. Check predictions alignment and thresholds.")

    coord_reg = make_coord_regressor(args.coord_reg)
    iou_reg   = make_iou_regressor(args.iou_reg)

    print("[fit] coordinate regressor...")
    coord_reg.fit(Xc, yc)
    print("[fit] IoU regressor...")
    iou_reg.fit(Xi, yi)

    # Save models
    try:
        import joblib
        joblib.dump(coord_reg, os.path.join(args.outdir, "coord_reg.joblib"))
        joblib.dump(iou_reg,   os.path.join(args.outdir, "iou_reg.joblib"))
    except Exception:
        pass

    # ---- TEST ----
    coco_te, images_te, _ = load_coco_gt(args.test_gt)
    models_te, per_model_te = load_predictions_per_model(args.test_preds, images_te)
    print("[test] models:", models_te)

    # IMPORTANT: model order at test must match train for feature layout
    if models_te != models_tr:
        print("[warn] test model list differs from train; reordering to match training order")
        models_te = models_tr
        per_model_te = {m: per_model_te.get(m, defaultdict(list)) for m in models_te}

    # Per-model baselines on TEST (simplified metrics)
    eval_single_models(coco_te, args.test_preds, images_te, split="test")

    # Ensemble inference
    preds = infer_split(
        images_by_id=images_te,
        models=models_te,
        per_model=per_model_te,
        coord_reg=coord_reg,
        iou_reg=iou_reg,
        min_neighbor_iou=args.min_neighbor_iou,
        anchor_score_thresh=args.anchor_score_thresh,
        nms_iou=args.nms_iou,
    )

    out_json = os.path.join(args.outdir, "learnable_fusion_preds.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"[OK] wrote predictions: {out_json}")

    # Ensemble metrics (simplified)
    ens_metrics = compute_pr_at_thresholds(coco_te, preds)
    print_metrics("Ensemble", ens_metrics)

    # (Optional) Also show full COCO summary for the ensemble
    print("\n=== COCO class-agnostic evaluation on TEST (Ensemble) ===")
    coco_eval_classagnostic(coco_te, preds)


if __name__ == "__main__":
    main()
