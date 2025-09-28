#!/usr/bin/env python3
import os
import json
import glob
import argparse
import numpy as np
from typing import Dict, Tuple, List
import cv2

def load_crop_coordinates(npz_path: str) -> Tuple[Dict[str, Tuple[int, int, int, int]], List[str]]:
    crop_data = np.load(npz_path, allow_pickle=True)
    img_dir = str(crop_data["img_list"])
    coords = crop_data["coordinates"]

    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    if len(files) != len(coords):
        raise ValueError(f"Mismatch: {len(files)} images vs {len(coords)} coordinates")

    fname_to_coords = {os.path.basename(p): tuple(map(int, c)) for p, c in zip(files, coords)}
    ordered_filenames = [os.path.basename(p) for p in files]
    return fname_to_coords, ordered_filenames


def load_gt_maps(annotation_json: str):
    with open(annotation_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    fname_to_id = {}
    id_to_meta = {}
    for img in coco["images"]:
        fid = int(img["id"])
        fname = img["file_name"]
        w = int(img["width"])
        h = int(img["height"])
        fname_to_id[fname] = fid
        id_to_meta[fid] = (fname, w, h)

    categories = coco.get("categories", [])
    return fname_to_id, id_to_meta, categories


def build_old_id_to_filename(ordered_filenames: List[str]) -> Dict[int, str]:
    return {i + 1: fname for i, fname in enumerate(ordered_filenames)}


def clip_bbox_xywh(x, y, w, h, W, H):
    x2 = x + w
    y2 = y + h
    x_cl = max(0.0, min(W, x))
    y_cl = max(0.0, min(H, y))
    x2_cl = max(0.0, min(W, x2))
    y2_cl = max(0.0, min(H, y2))
    w_new = x2_cl - x_cl
    h_new = y2_cl - y_cl
    clipped = (x_cl != x) or (y_cl != y) or (x2_cl != x2) or (y2_cl != y2)
    dropped = (w_new <= 0.0) or (h_new <= 0.0)
    return [x_cl, y_cl, w_new, h_new], clipped, dropped


def correct_and_build_coco(
    input_json: str,
    output_json: str,
    fname_to_coords: Dict[str, Tuple[int, int, int, int]],
    ordered_filenames: List[str],
    fname_to_id: Dict[str, int],
    id_to_meta: Dict[int, Tuple[str, int, int]],
    categories: List[Dict],
    fname_to_size: Dict[str, Tuple[int, int]] = None,
):
    with open(input_json, "r", encoding="utf-8") as f:
        preds = json.load(f)

    old_to_fname = build_old_id_to_filename(ordered_filenames)
    # print(old_to_fname)
    gt_id_to_fname = {fid: meta[0] for fid, meta in id_to_meta.items()}

    coco_out = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    if fname_to_size is None:
        for fid, (fname, w, h) in id_to_meta.items():
            coco_out["images"].append({
                "id": fid,
                "file_name": fname,
                "width": w,
                "height": h,
            })
    else:
        print("Using provided image sizes for output JSON")
        for fid in old_to_fname:
            fname = old_to_fname[fid]
            w, h = fname_to_size.get(fname, (0, 0))
            coco_out["images"].append({
                "id": fid,
                "file_name": fname,
                "width": w,
                "height": h,
            })

    ann_id = 1
    stats = {"warn_no_map": 0, "clipped": 0, "dropped": 0}

    for pred in preds:
        pid = pred.get("image_id")
        fname = None
        if isinstance(pid, (int, np.integer)) and pid in old_to_fname:
            # print('Warning: Using old ID mapping for image_id:', pid)
            fname = old_to_fname[pid]
        elif isinstance(pid, (int, np.integer)) and pid in gt_id_to_fname:
            # print('Warning: Using GT ID mapping for image_id:', pid)
            fname = gt_id_to_fname[pid]
        else:
            stats["warn_no_map"] += 1
            continue

        if fname not in fname_to_id:
            # print(f"Warning: No GT mapping for filename: {fname}")
            true_id = pid
            # print(f"Using original image_id: {true_id}")
            W, H = fname_to_size.get(fname, (0, 0))
        else:
            true_id = fname_to_id[fname]
            _, W, H = id_to_meta[true_id]

        x0, y0, _, _ = fname_to_coords.get(fname, (0, 0, 0, 0))
        x, y, w, h = pred["bbox"]
        x_s, y_s = x + x0, y + y0

        
        bbox_new, clipped, dropped = clip_bbox_xywh(x_s, y_s, w, h, W, H)
        if clipped:
            stats["clipped"] += 1
        if dropped:
            stats["dropped"] += 1
            continue

        area = bbox_new[2] * bbox_new[3]
        coco_out["annotations"].append({
            "id": ann_id,
            "image_id": true_id,
            "category_id": pred.get("category_id"),
            "bbox": bbox_new,
            "area": area,
            "iscrowd": 0,
            "segmentation": [],
            "score": float(pred.get("score", 1.0)),  # preserve score
        })
        ann_id += 1

    coco_out["images"].sort(key=lambda x: x["id"])
    coco_out["annotations"].sort(key=lambda x: x["id"])

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_out, f)

    print(f"Written corrected COCO JSON: {output_json}")
    if stats["warn_no_map"] > 0:
        print(f"WARNING: {stats['warn_no_map']} predictions had no valid image mapping.")
    if stats["clipped"] > 0:
        print(f"INFO: {stats['clipped']} boxes clipped to image boundaries.")
    if stats["dropped"] > 0:
        print(f"INFO: {stats['dropped']} degenerate boxes dropped.")


def batch_process(json_dir: str, npz_path: str, annotation_json: str, output_dir: str, image_path: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    fname_to_coords, ordered_filenames = load_crop_coordinates(npz_path)
    # print(ordered_filenames)
    fname_to_id, id_to_meta, categories = load_gt_maps(annotation_json)
    # print(f"{fname_to_id=}")
    print("loading image sizes...")
    fname_to_size = {}
    for fname in ordered_filenames:
        img_file = os.path.join(image_path, fname)
        if os.path.isfile(img_file):
            img = cv2.imread(img_file)
            if img is not None:
                h, w = img.shape[:2]
                fname_to_size[fname] = (w, h)
            else:
                print(f"Warning: Unable to read image {img_file}")
        else:
            print(f"Warning: Image file {img_file} does not exist")
    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue
        inp = os.path.join(json_dir, fname)
        out = os.path.join(output_dir, fname)
        correct_and_build_coco(
            inp, out, fname_to_coords, ordered_filenames, fname_to_id, id_to_meta, categories, fname_to_size
        )


def main():
    parser = argparse.ArgumentParser(description="Correct cropped COCO predictions to full COCO format")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_path", type=str, help="Path to the directory containing original images", required=False)
    args = parser.parse_args()

    batch_process(args.json_dir, args.npz_path, args.annotation_json, args.output_dir, args.img_path)


if __name__ == "__main__":
    main()
