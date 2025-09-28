#!/usr/bin/env python3
"""
Inference-only script for the learnable, class-agnostic box fusion.

Outputs a FULL COCO-style JSON dict:
{
  "info": {...},
  "licenses": [...],
  "images": [...],
  "categories": [...],
  "annotations": [...]   # ensemble predictions as detection annotations
}

Each prediction annotation has:
  - id: unique int
  - image_id: numeric id aligned to images
  - category_id: BI-RADS category (or fixed id)
  - bbox: [x, y, w, h] integers, clipped to image size if known
  - score: float rounded to 4 decimals
  - iscrowd: 0
  - area: w*h (integer)

Category/source precedence:
  1) --images_json (if provided) supplies info/licenses/images/categories
  2) otherwise, the first predictions JSON that contains those sections
  3) otherwise, we synthesize minimal sections (single category if needed)

Author: you :)
"""

import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import joblib

# Optional: only to be import-safe if xgboost isn't installed
try:
    from xgboost import XGBRegressor  # noqa: F401
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


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
    if len(dets) == 0:
        return []
    boxes = np.stack([xywh_to_xyxy(b) for b in dets], axis=0)
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

def _read_predictions_json(pred_path: str) -> Tuple[List[dict], Optional[dict]]:
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        anns = data.get("annotations")
        if isinstance(anns, list):
            return anns, data
    raise ValueError(f"Unsupported predictions format: {pred_path}")

def _extract_scaffold_from_dict(d: dict) -> Dict[str, Any]:
    """Pull out COCO scaffold pieces if present."""
    return {
        "info": d.get("info"),
        "licenses": d.get("licenses"),
        "images": d.get("images"),
        "categories": d.get("categories"),
    }

def build_coco_scaffold(pred_paths: List[str], images_json: Optional[str]) -> Dict[str, Any]:
    """
    Build base COCO dict: info/licenses/images/categories.
    Priority:
      1) --images_json
      2) first pred JSON with those keys
      3) synthesize minimal scaffold (info/licenses empty, images unknown)
    """
    scaffold = {"info": None, "licenses": None, "images": None, "categories": None}

    # 1) images_json
    if images_json:
        with open(images_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "images" not in data:
            raise ValueError("--images_json must be a COCO dict containing an 'images' list")
        sc = _extract_scaffold_from_dict(data)
        for k in scaffold:
            scaffold[k] = sc.get(k) if sc.get(k) is not None else scaffold.get(k)

    # 2) fall back to first pred with sections
    if not scaffold["images"] or not scaffold["categories"] or scaffold["info"] is None or scaffold["licenses"] is None:
        for p in pred_paths:
            try:
                _, raw = _read_predictions_json(p)
            except Exception:
                continue
            if isinstance(raw, dict):
                sc = _extract_scaffold_from_dict(raw)
                for k in scaffold:
                    if scaffold[k] is None and sc.get(k) is not None:
                        scaffold[k] = sc.get(k)
            # Stop if all are filled
            if all(scaffold[k] is not None for k in scaffold):
                break

    # 3) synthesize missing pieces
    if scaffold["info"] is None:
        scaffold["info"] = {
            "description": "Ensemble predictions",
            "version": "1.0",
        }
    if scaffold["licenses"] is None:
        scaffold["licenses"] = []

    # Ensure images list exists
    if scaffold["images"] is None:
        scaffold["images"] = []

    # Ensure categories list
    if scaffold["categories"] is None:
        scaffold["categories"] = []

    # Deduplicate images by id, keep the latest occurrence
    seen = {}
    for img in scaffold["images"]:
        if "id" in img:
            seen[int(img["id"])] = img
    scaffold["images"] = [seen[i] for i in sorted(seen.keys())]

    # Categories: ensure unique by id
    if scaffold["categories"]:
        cat_by_id = {}
        for c in scaffold["categories"]:
            if "id" in c:
                cat_by_id[int(c["id"])] = c
        scaffold["categories"] = [cat_by_id[i] for i in sorted(cat_by_id.keys())]

    return scaffold

def build_images_map(scaffold: Dict[str, Any]) -> Dict[int, dict]:
    return {int(img["id"]): img for img in scaffold.get("images", []) if "id" in img}

def build_categories_map(scaffold: Dict[str, Any]) -> Dict[int, dict]:
    return {int(cat["id"]): cat for cat in scaffold.get("categories", []) if "id" in cat}

def _maybe_remap_ids_by_filename(
    dets: List[dict], pred_dict: Any, images_by_id: Dict[int, dict]
) -> List[dict]:
    if not images_by_id:
        return dets
    gt_ids = set(images_by_id.keys())
    if all(int(d.get("image_id")) in gt_ids for d in dets):
        return dets
    if not (isinstance(pred_dict, dict) and "images" in pred_dict):
        return dets
    pid2fname = {int(img["id"]): img["file_name"] for img in pred_dict.get("images", []) if "id" in img}
    fname2gtid = {meta.get("file_name"): iid for iid, meta in images_by_id.items() if "file_name" in meta}
    fixed = []
    for d in dets:
        iid = int(d.get("image_id"))
        if iid in gt_ids:
            fixed.append(d); continue
        fname = pid2fname.get(iid)
        if fname is not None and fname in fname2gtid:
            nd = dict(d)
            nd["image_id"] = int(fname2gtid[fname])
            fixed.append(nd)
        # else: drop if cannot map
    return fixed

def load_predictions_per_model(
    pred_paths: List[str],
    images_by_id: Dict[int, dict]
):
    """
    Returns:
      models: List[str]
      per_model: Dict[model -> Dict[image_id(int) -> List[(bbox, score, cat)]]]
      any_categories_used: set of category ids present in predictions
    """
    per_model = {}
    models = []
    cats_used = set()
    for p in pred_paths:
        model = os.path.splitext(os.path.basename(p))[0]
        dets, raw = _read_predictions_json(p)
        dets = _maybe_remap_ids_by_filename(dets, raw, images_by_id)
        imgmap = defaultdict(list)
        for d in dets:
            iid = int(d.get("image_id"))
            bbox = d.get("bbox", [0,0,0,0])
            score = float(d.get("score", 1.0))
            cat = d.get("category_id", None)
            if cat is not None:
                cats_used.add(int(cat))
            imgmap[iid].append((bbox, score, cat))
        models.append(model)
        per_model[model] = imgmap
    return models, per_model, cats_used


# ---------------------------
# Feature building
# ---------------------------

def iou_anchor(anchor_xyxy, b_xywh):
    return iou_xyxy(anchor_xyxy, xywh_to_xyxy(b_xywh))

def best_match_to_anchor(anchor_xyxy, cand_xywh_list: List[Tuple[List[float], float, Any]]):
    if not cand_xywh_list:
        return None, 0.0, 0.0, None
    best_iou = -1.0
    best = None
    best_cat = None
    best_score = 0.0
    for (b, s, c) in cand_xywh_list:
        iou = iou_anchor(anchor_xyxy, b)
        if iou > best_iou:
            best_iou = iou
            best = b
            best_score = float(s)
            best_cat = c
    return best, best_score, float(max(0.0, best_iou)), best_cat

def build_feature_vector(
    anchor_xywh, anchor_score, anchor_model_idx, model_names,
    per_model_boxes_for_image: Dict[str, List[Tuple[List[float], float, Any]]],
    img_w: int, img_h: int,
    min_neighbor_iou: float = 0.05
) -> np.ndarray:
    W, H = float(img_w), float(img_h)
    ax, ay, aw, ah = anchor_xywh
    feat = []

    for m in model_names:
        cand = per_model_boxes_for_image.get(m, [])
        b, s, iou, _ = best_match_to_anchor(xywh_to_xyxy(anchor_xywh), cand)
        if b is None or iou < min_neighbor_iou:
            feat.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            bx, by, bw, bh = b
            feat.extend([
                bx / max(W,1e-6), by / max(H,1e-6),
                bw / max(W,1e-6), bh / max(H,1e-6),
                s, iou, 1.0
            ])

    feat.extend([ax / max(W,1e-6), ay / max(H,1e-6), aw / max(W,1e-6), ah / max(H,1e-6), float(anchor_score)])

    onehot = [0.0] * len(model_names)
    if 0 <= anchor_model_idx < len(model_names):
        onehot[anchor_model_idx] = 1.0
    feat.extend(onehot)

    return np.array(feat, dtype=np.float32)


# ---------------------------
# Utils
# ---------------------------

def clip_and_round_xywh(x, y, w, h, W=None, H=None):
    """Round to ints and clip to image bounds if known."""
    if W is not None and H is not None:
        x = max(0.0, min(float(x), float(W)))
        y = max(0.0, min(float(y), float(H)))
        w = max(0.0, min(float(w), float(W) - float(x)))
        h = max(0.0, min(float(h), float(H) - float(y)))
    xi = int(round(x)); yi = int(round(y))
    wi = int(round(w)); hi = int(round(h))
    if W is not None and H is not None:
        wi = max(0, min(wi, int(W) - xi))
        hi = max(0, min(hi, int(H) - yi))
    return xi, yi, wi, hi


# ---------------------------
# Inference (returns prediction annotations)
# ---------------------------

def infer_split(
    images_by_id: Dict[int, dict],
    models: List[str],
    per_model: Dict[str, Dict[int, List[Tuple[List[float], float, Any]]]],
    coord_reg,
    iou_reg,
    min_neighbor_iou: float = 0.05,
    anchor_score_thresh: float = 0.0,
    nms_iou: float = 0.5,
    category_mode: str = "fixed",
    fixed_category_id: int = 1,
) -> List[dict]:
    """
    Returns list of COCO annotations (predictions).
    """
    outputs = []
    ann_id = 1

    # union of image ids across scaffold and predictions
    all_image_ids = set(images_by_id.keys())
    if not all_image_ids:
        for m in models:
            all_image_ids |= set(per_model.get(m, {}).keys())

    for iid in sorted(all_image_ids):
        meta = images_by_id.get(iid, {})
        W = meta.get("width", None)
        H = meta.get("height", None)

        feats = []
        anchors_meta = []

        # collect anchors from all models
        for mi, m in enumerate(models):
            dets = per_model.get(m, {}).get(iid, [])
            for (b, s, c) in dets:
                if s < anchor_score_thresh:
                    continue
                feat = build_feature_vector(
                    anchor_xywh=b, anchor_score=s, anchor_model_idx=mi,
                    model_names=models,
                    per_model_boxes_for_image={mm: per_model.get(mm, {}).get(iid, []) for mm in models},
                    img_w=int(W) if W is not None else 1,
                    img_h=int(H) if H is not None else 1,
                    min_neighbor_iou=min_neighbor_iou
                )
                feats.append(feat)
                anchors_meta.append((mi, c, b, s))

        if not feats:
            continue

        X = np.stack(feats, axis=0)
        ybox = coord_reg.predict(X)               # normalized [N,4]
        yiou = iou_reg.predict(X).reshape(-1)     # [N] (0..1)

        # denormalize to pixels
        boxes = []
        for row in ybox:
            if W is not None and H is not None:
                x = max(0.0, min(1.0, float(row[0]))) * float(W)
                y = max(0.0, min(1.0, float(row[1]))) * float(H)
                w = max(0.0, min(1.0, float(row[2]))) * float(W)
                h = max(0.0, min(1.0, float(row[3]))) * float(H)
            else:
                x, y, w, h = [float(v) for v in row]
            boxes.append([x, y, w, h])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.clip(yiou, 0.0, 1.0)

        keep = nms_xywh(boxes, scores, iou_thr=nms_iou)

        for k in keep:
            # choose category
            cat_id = fixed_category_id
            if category_mode == "pass_through":
                _, src_cat, _, _ = anchors_meta[k]
                if src_cat is not None:
                    cat_id = int(src_cat)
            elif category_mode == "majority":
                refined_xyxy = xywh_to_xyxy(boxes[k])
                votes = []
                for m in models:
                    cand = per_model.get(m, {}).get(iid, [])
                    best_c = None
                    best_i = -1.0
                    for (b, s, c) in cand:
                        iou = iou_xyxy(refined_xyxy, xywh_to_xyxy(b))
                        if iou > best_i:
                            best_i = iou; best_c = c
                    if best_c is not None and best_i >= min_neighbor_iou:
                        votes.append(int(best_c))
                if votes:
                    cat_id = Counter(votes).most_common(1)[0][0]

            x, y, w, h = boxes[k].tolist()
            xi, yi, wi, hi = clip_and_round_xywh(x, y, w, h, W=W, H=H)
            if scores[k] > 0.01:
                print(scores[k])
                outputs.append({
                    "id": ann_id,
                    "image_id": int(iid),
                    "category_id": int(cat_id),
                    "bbox": [xi, yi, wi, hi],
                    "score": float(round(float(scores[k]*3), 4)),
                    "iscrowd": 0,
                    "area": int(wi * hi),
                })
                ann_id += 1

    return outputs


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser("Inference for learnable class-agnostic fusion (FULL COCO dict output)")
    ap.add_argument("--ensemble_dir", required=True, help="Directory with coord_reg.joblib and iou_reg.joblib")
    ap.add_argument("--preds", required=True, nargs="+", help="Per-model prediction JSONs for the new dataset")
    ap.add_argument("--images_json", default=None, help="Optional COCO JSON to source info/licenses/images/categories")
    ap.add_argument("--out_json", required=True, help="Output COCO JSON path")
    ap.add_argument("--min_neighbor_iou", type=float, default=0.05)
    ap.add_argument("--anchor_score_thresh", type=float, default=0.0)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--category_mode", choices=["fixed","pass_through","majority"], default="fixed",
                    help="How to choose category_id for predictions")
    ap.add_argument("--fixed_category_id", type=int, default=1,
                    help="Used when category_mode='fixed'")
    args = ap.parse_args()

    # Load trained regressors
    coord_path = os.path.join(args.ensemble_dir, "coord_reg.joblib")
    iou_path   = os.path.join(args.ensemble_dir, "iou_reg.joblib")
    if not (os.path.isfile(coord_path) and os.path.isfile(iou_path)):
        raise FileNotFoundError(f"Could not find trained models in {args.ensemble_dir}")
    coord_reg = joblib.load(coord_path)
    iou_reg   = joblib.load(iou_path)

    # Build scaffold (info/licenses/images/categories)
    scaffold = build_coco_scaffold(args.preds, args.images_json)
    images_by_id = build_images_map(scaffold)
    categories_by_id = build_categories_map(scaffold)

    # Load per-model detections
    models, per_model, cats_used = load_predictions_per_model(args.preds, images_by_id)
    print("[info] models:", models)

    # If no categories exist in scaffold, create them
    if not scaffold["categories"]:
        if args.category_mode == "fixed":
            scaffold["categories"] = [{"id": int(args.fixed_category_id), "name": "lesion"}]
        else:
            # Build from used ids in predictions
            if not cats_used:
                scaffold["categories"] = [{"id": 1, "name": "lesion"}]
            else:
                scaffold["categories"] = [{"id": int(cid), "name": f"class_{int(cid)}"} for cid in sorted(cats_used)]
        categories_by_id = build_categories_map(scaffold)

    # Run ensemble inference -> annotations
    annotations = infer_split(
        images_by_id=images_by_id,
        models=models,
        per_model=per_model,
        coord_reg=coord_reg,
        iou_reg=iou_reg,
        min_neighbor_iou=args.min_neighbor_iou,
        anchor_score_thresh=args.anchor_score_thresh,
        nms_iou=args.nms_iou,
        category_mode=args.category_mode,
        fixed_category_id=args.fixed_category_id,
    )

    # Assemble full COCO dict
    coco_out = {
        "info": scaffold.get("info", {}),
        "licenses": scaffold.get("licenses", []),
        "images": scaffold.get("images", []),
        "categories": scaffold.get("categories", []),
        "annotations": annotations,
    }

    # Write pretty JSON
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote: {args.out_json}")
    print(f"[stats] images={len(coco_out['images'])}, categories={len(coco_out['categories'])}, annotations={len(annotations)}")


if __name__ == "__main__":
    main()
