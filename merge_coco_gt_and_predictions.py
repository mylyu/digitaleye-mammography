#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, List, Any, Tuple


def _safe_float_bbox(bbox):
    # Ensure xywh floats
    x, y, w, h = bbox
    return [float(x), float(y), float(w), float(h)]


def _compute_area(bbox):
    return float(bbox[2]) * float(bbox[3])


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pred_annotations_from_file(pred_data: Any) -> List[dict]:
    """
    Accept either:
      - full COCO dict with "annotations" (preferred)
      - or a list of prediction dicts (legacy)
    """
    if isinstance(pred_data, dict) and "annotations" in pred_data:
        return list(pred_data["annotations"])
    if isinstance(pred_data, list):
        return pred_data
    raise ValueError("Prediction file must be a list or a COCO-style dict with 'annotations'.")


def _coerce_coco_annotation(
    ann: dict,
    gt_img_ids: set,
    gt_cat_ids: set,
    source_tag: str,
    model_tag: str = None,
) -> Tuple[dict, List[str]]:
    """
    Normalize a single annotation to COCO detection spec.
    - Force iscrowd=0
    - Ensure bbox xywh floats
    - Ensure area present
    - Keep/assign score (preds)
    - Tag with 'source' and optional 'model'
    Return (normalized_ann, warnings)
    """
    warnings = []

    image_id = ann.get("image_id")
    category_id = ann.get("category_id")

    if image_id not in gt_img_ids:
        warnings.append(f"annotation image_id {image_id} not in GT images")
    if category_id not in gt_cat_ids:
        warnings.append(f"annotation category_id {category_id} not in GT categories")

    bbox = _safe_float_bbox(ann.get("bbox", [0, 0, 0, 0]))

    norm = {
        # 'id' will be re-assigned later
        "image_id": int(image_id) if image_id is not None else None,
        "category_id": int(category_id) if category_id is not None else None,
        "bbox": bbox,
        "area": float(ann.get("area", _compute_area(bbox))),
        "iscrowd": 0,
        "segmentation": ann.get("segmentation", []),
        "source": source_tag,
    }

    # Preserve score for predictions; for GT you can keep 1.0 to ease filtering
    if "score" in ann:
        norm["score"] = float(ann["score"])
    else:
        # give GT or score-less preds a default score so you can filter in viewers
        norm["score"] = 1.0 if source_tag == "gt" else 1.0

    if model_tag:
        norm["model"] = model_tag

    return norm, warnings


def merge_coco(
    gt_json_path: str,
    pred_json_paths: List[str],
    output_json_path: str,
) -> None:
    # --- Load GT ---
    gt = _load_json(gt_json_path)

    images = gt.get("images", [])
    categories = gt.get("categories", [])
    gt_annotations = gt.get("annotations", [])
    info = gt.get("info", {})
    licenses = gt.get("licenses", [])

    # Build quick maps/sets
    gt_img_ids = set(int(img["id"]) for img in images)
    gt_cat_ids = set(int(cat["id"]) for cat in categories)

    # Normalize + tag GT anns
    merged_annotations: List[dict] = []
    warn_count = {"img_miss": 0, "cat_miss": 0}

    for ann in gt_annotations:
        norm, warns = _coerce_coco_annotation(ann, gt_img_ids, gt_cat_ids, source_tag="gt")
        for w in warns:
            if "image_id" in w: warn_count["img_miss"] += 1
            if "category_id" in w: warn_count["cat_miss"] += 1
        merged_annotations.append(norm)

    # --- Load predictions (each file) ---
    for ppath in pred_json_paths:
        pdata = _load_json(ppath)
        anns = _pred_annotations_from_file(pdata)
        # model tag derived from filename
        model_tag = os.path.splitext(os.path.basename(ppath))[0]

        for ann in anns:
            norm, warns = _coerce_coco_annotation(ann, gt_img_ids, gt_cat_ids,
                                                  source_tag="pred", model_tag=model_tag)
            for w in warns:
                if "image_id" in w: warn_count["img_miss"] += 1
                if "category_id" in w: warn_count["cat_miss"] += 1
            merged_annotations.append(norm)

    # Re-index annotation IDs uniquely
    for i, ann in enumerate(merged_annotations, start=1):
        ann["id"] = i

    # Compose final merged COCO
    merged = {
        "info": info,        # keep GT info
        "licenses": licenses,
        "images": images,    # GT images (IDs must match)
        "annotations": merged_annotations,
        "categories": categories,
    }

    # Save
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    print(f"[OK] Wrote merged COCO: {output_json_path}")

    # Warnings summary
    if warn_count["img_miss"] > 0:
        print(f"[WARN] {warn_count['img_miss']} annotations referenced image_ids not in GT images.")
    if warn_count["cat_miss"] > 0:
        print(f"[WARN] {warn_count['cat_miss']} annotations referenced category_ids not in GT categories.")
    print(f"[INFO] Total annotations in merged: {len(merged_annotations)}")


def main():
    ap = argparse.ArgumentParser(description="Merge GT + prediction COCO JSONs into one, tagged with sources.")
    ap.add_argument("--gt_json", required=True, help="Path to GT COCO JSON")
    ap.add_argument("--pred_json", required=True, nargs="+", help="One or more prediction JSONs")
    ap.add_argument("--output_json", required=True, help="Output merged JSON")
    args = ap.parse_args()

    merge_coco(args.gt_json, args.pred_json, args.output_json)


if __name__ == "__main__":
    main()
