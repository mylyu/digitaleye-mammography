#!/usr/bin/env python3
"""
Post-process COCO predictions for BI-RADS relabeling.

Rules:
  - Input "category_id": 1 => Malign, 2 => Benign.
  - Take the top <top_pct> by score within Malign -> relabel to category_id=4 (BI-RADS-4).
  - Take the top <top_pct> by score within Benign -> keep category_id=2 (BI-RADS-2).
  - All other detections (including categories other than 1/2) -> category_id=0 (BI-RADS-0).
  - For images with no detections at all in the INPUT file -> add one centered, large box with category_id=1 (BI-RADS-1).

Other behaviors:
  - New fallback box uses a fraction of image size (--fallback_bbox_ratio) and is centered.
  - Keeps info/licenses/images/categories as-is; rewrites annotations accordingly.
  - Pretty-prints output JSON.

Usage:
  python postprocess_birads.py \
      --in /path/to/input.json \
      --out /path/to/output.json \
      --top_pct 0.30 \
      --fallback_bbox_ratio 0.6
"""

import argparse
import json
import math
from typing import Dict, List, Any

def parse_args():
    ap = argparse.ArgumentParser("BI-RADS COCO post-processor")
    ap.add_argument("--in", dest="inp", required=True, help="Input COCO JSON (with info/licenses/images/categories/annotations)")
    ap.add_argument("--out", dest="out", required=True, help="Output COCO JSON")
    ap.add_argument("--top_pct", type=float, default=0.30, help="Top fraction (0-1) to keep as high-score within each of Malign/Benign")
    ap.add_argument("--fallback_bbox_ratio", type=float, default=0.60, help="Width/height fraction for centered fallback box when an image had no detections")
    ap.add_argument("--score_key", type=str, default="score", help="Score field name inside annotations")
    return ap.parse_args()

def _top_k_indices_by_score(items: List[Dict[str, Any]], k: int, score_key: str) -> set:
    if k <= 0 or not items:
        return set()
    # Stable sort by score desc; if no score, treat as 0
    sorted_idx = sorted(range(len(items)),
                        key=lambda i: float(items[i].get(score_key, 0.0)),
                        reverse=True)
    k = min(k, len(sorted_idx))
    return set(sorted_idx[:k])

def _centered_box(w: int, h: int, frac: float):
    frac = max(0.05, min(0.95, float(frac)))
    bw = int(round(w * frac))
    bh = int(round(h * frac))
    x = max(0, (w - bw) // 2)
    y = max(0, (h - bh) // 2)
    # ensure stays in bounds
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    return [int(x), int(y), int(bw), int(bh)]

def main():
    args = parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic structure checks (soft)
    for key in ("info", "licenses", "images", "categories", "annotations"):
        if key not in data:
            raise ValueError(f"Input JSON missing '{key}' key. This script expects a full COCO-style dict.")

    images = data["images"]
    anns_in: List[Dict[str, Any]] = data["annotations"]

    # Build per-image detection existence from INPUT (for the fallback rule)
    has_input_det = {img["id"]: False for img in images}
    for a in anns_in:
        iid = a.get("image_id")
        if iid in has_input_det:
            has_input_det[iid] = True

    # Separate malignant (cat=1) and benign (cat=2) from INPUT
    malign = []
    benign = []
    others = []  # any other category id in the input (we'll send to BI-RADS-0)
    for ann in anns_in:
        cid = int(ann.get("category_id", -1))
        if cid == 1:
            malign.append(ann)
        elif cid == 2:
            benign.append(ann)
        else:
            others.append(ann)

    # Compute top-k within each group
    def topk_count(n: int, pct: float) -> int:
        return max(0, int(math.ceil(n * max(0.0, min(1.0, pct)))))

    k_malign = topk_count(len(malign), args.top_pct)
    k_benign = topk_count(len(benign), args.top_pct)

    top_malign_idx = _top_k_indices_by_score(malign, k_malign, args.score_key)
    top_benign_idx = _top_k_indices_by_score(benign, k_benign, args.score_key)
    print(sorted(benign, key=lambda d: d[args.score_key]))

    # Rewrite annotations according to the rules
    new_anns: List[Dict[str, Any]] = []
    for i, ann in enumerate(malign):
        ann = dict(ann)
        if i in top_malign_idx:
            ann["category_id"] = 4  # BI-RADS-4
        else:
            ann["category_id"] = 0  # BI-RADS-0
        new_anns.append(ann)

    for i, ann in enumerate(benign):
        ann = dict(ann)
        if i not in top_benign_idx:
            ann["category_id"] = 2  # BI-RADS-2
            new_anns.append(ann)
        else: #
            print('skip this benign because too high')
        

    for ann in others:
        ann = dict(ann)
        ann["category_id"] = 0  # BI-RADS-0
        new_anns.append(ann)

    # Add fallback detections for images with no detections in INPUT
    # Pick next annotation id
    max_id = 0
    for a in new_anns:
        try:
            max_id = max(max_id, int(a.get("id", 0)))
        except Exception:
            pass
    next_id = max_id + 1

    # Build quick image meta lookup
    img_meta = {img["id"]: img for img in images}

    for img in images:
        iid = img["id"]
        if not has_input_det.get(iid, False):
            w, h = int(img.get("width", 0)), int(img.get("height", 0))
            bbox = _centered_box(w, h, args.fallback_bbox_ratio)
            a = {
                "id": next_id,
                "image_id": iid,
                "category_id": 1,      # BI-RADS-1 fallback
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "score": 0.01,         # low confidence placeholder
            }
            # Optional COCO fields if you want them
            a["iscrowd"] = 0
            a["area"] = int(bbox[2]) * int(bbox[3])
            new_anns.append(a)
            next_id += 1

    # Write back as COCO dict (pretty)
    out_dict = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "categories": data.get("categories", []),
        "annotations": new_anns,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote post-processed predictions to: {args.out}")
    print(f"  Malign: {len(malign)} (top {k_malign})  |  Benign: {len(benign)} (top {k_benign})  |  Others: {len(others)}")
    print(f"  Images with no input detections -> added fallbacks: {sum(1 for k,v in has_input_det.items() if not v)}")

if __name__ == "__main__":
    main()
