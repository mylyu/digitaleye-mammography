import os
import cv2
import glob
import json
import argparse
import numpy as np
from utils.all_utils import *
import pickle
import warnings
import time
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _slugify(value: str) -> str:
    """Return a filesystem-friendly version of the provided string."""

    safe_chars = []
    for character in value.strip():
        if character.isalnum():
            safe_chars.append(character.lower())
        elif character in {"_", "-"}:
            safe_chars.append(character)
        elif character.isspace():
            safe_chars.append("_")
    return "".join(safe_chars) or "predictions"


def _prepare_coco_resources(
    img_list: Sequence[str],
    label_names: Sequence[str],
    coco_gt_path: Optional[str] = None,
) -> Tuple[Optional[COCO], Dict[str, int], List[int]]:
    """Create helper structures for exporting and evaluating COCO predictions."""

    image_id_map: Dict[str, int] = {}
    coco_gt: Optional[COCO] = None
    category_ids: List[int] = []

    if coco_gt_path:
        if not os.path.isfile(coco_gt_path):
            raise FileNotFoundError(f"COCO annotation file not found: {coco_gt_path}")
        coco_gt = COCO(coco_gt_path)
        for img_id, img_info in coco_gt.imgs.items():
            image_id_map[os.path.basename(img_info["file_name"])] = img_id
        categories = coco_gt.loadCats(coco_gt.getCatIds())
        cat_name_map = {cat["name"].lower(): cat["id"] for cat in categories}
        for label in label_names:
            key = label.lower()
            if key not in cat_name_map:
                raise KeyError(
                    f"Label '{label}' could not be matched to a category in the COCO annotations."
                )
            category_ids.append(cat_name_map[key])
    else:
        # Create a deterministic mapping based on the given image order and label order.
        for index, img_path in enumerate(img_list, start=1):
            image_id_map[os.path.basename(img_path)] = index
        category_ids = list(range(1, len(label_names) + 1))

    return coco_gt, image_id_map, category_ids


def _build_coco_predictions(
    img_list: Sequence[str],
    results: Sequence[Sequence[np.ndarray]],
    image_id_map: Dict[str, int],
    category_ids: Sequence[int],
) -> List[Dict[str, float]]:
    """Convert model predictions to COCO detection format."""

    coco_predictions: List[Dict[str, float]] = []
    for img_path, detections in zip(img_list, results):
        image_key = os.path.basename(img_path)
        if image_key not in image_id_map:
            raise KeyError(
                f"Image '{image_key}' is missing from the COCO annotations and cannot be exported."
            )
        image_id = image_id_map[image_key]
        for class_index, class_detections in enumerate(detections):
            category_id = int(category_ids[class_index])
            for det in class_detections:
                x_min, y_min, x_max, y_max = det[:4]
                score = float(det[4]) if det.shape[0] >= 5 else 1.0
                width = max(float(x_max - x_min), 0.0)
                height = max(float(y_max - y_min), 0.0)
                coco_predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [float(x_min), float(y_min), width, height],
                        "score": score,
                    }
                )
    return coco_predictions


def _compute_coco_metrics(
    coco_gt: COCO,
    coco_predictions: List[Dict[str, float]],
    iou_threshold: float,
) -> Dict[str, float]:
    """Evaluate predictions using COCO metrics."""

    if not coco_predictions:
        # COCO expects an empty list to still be a valid result file.
        coco_dt = coco_gt.loadRes([])
    else:
        coco_dt = coco_gt.loadRes(coco_predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([iou_threshold])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precision_values = coco_eval.eval.get("precision")
    recall_values = coco_eval.eval.get("recall")

    precision = float("nan")
    recall = float("nan")

    if precision_values is not None:
        valid_precision = precision_values[precision_values > -1]
        if valid_precision.size > 0:
            precision = float(valid_precision.mean())

    if recall_values is not None:
        valid_recall = recall_values[recall_values > -1]
        if valid_recall.size > 0:
            recall = float(valid_recall.mean())

    ap = float(coco_eval.stats[0]) if coco_eval.stats.size > 0 else float("nan")

    return {"AP": ap, "precision": precision, "recall": recall}


# git ignore dosyasi olusturup results komple ignore, 
start_time = time.time()
print('PROCESSES STARTED')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model_enum', nargs='+', help='Example: --model_enum 0 or 0 1 2, Purpose: 0: ATSS, 1: Cascade R-CNN, 2: DEFORMABLE DETR, 3: DETR, 4: DOUBLE HEAD R-CNN, 5: DYNAMIC R-CNN, 6: FASTER R-CNN, 7: FCOS, 8: RETINA NET, 9: VARIFOCAL NET, 10: YOLOv3', required=True)
parser.add_argument('--device', type=str, help='Example: --device cuda:0', required=False, default='cpu')
parser.add_argument('--classify_mass', type=str, help='Example: --classify_mass True or False, Purpose: It is for doing classificiation', required=False, default='True') 
parser.add_argument('--segment_breast', type=str , help='Example: --segment_breast True or False, Purpose: It is for applying breast segmentation model', required=False, default='True') 
parser.add_argument('--enable_ensemble', type=str, help='Example: --enable_ensemble True or False, Purpose: It is for applying ensemble strategy to detections', required=False, default='False')
parser.add_argument('--img_path', type=str, help='Example: --PATH, Purpose: It is for getting image folder path', required=True, default=None)
parser.add_argument('--annotation_path', type=str, help='Example: --PATH, Purpose: It is for getting annotation .txt file path', required=False, default=None)
parser.add_argument('--nms_iou_threshold', type=float, help='Example: --nms_iou_threshold 0.1', required=False, default=0.1)
parser.add_argument('--confidence_threshold', type=float, help='Example: --confidence_threshold 0.2', required=False, default=0.2)
parser.add_argument('--ap_threshold', type=float, help='Example: --ap_threshold 0.1', required=False, default=0.1)
parser.add_argument(
    '--coco_output_dir',
    type=str,
    help='Directory for saving COCO formatted predictions (defaults to results_dir/coco_predictions)',
    required=False,
    default=None,
)
parser.add_argument('--gt_coco_path', type=str, help='Ground truth COCO annotations for evaluation', required=False, default=None)

args = parser.parse_args()

models_path = 'models/'
if not os.path.exists(models_path):
    os.mkdir(models_path)

if len(glob.glob(os.path.join(args.img_path, '*.[pP][nN][gG]'))) == 0:
    print(args.img_path, 'not include any images... You must give image folder which contains images...')
    parser.print_help()
    exit(-1)

if len(args.model_enum) == 1 and args.enable_ensemble == 'True':
    print('You must give more than one model for applying ensemble strategy')
    parser.print_help()
    exit(-1)

model_names = args.model_enum
device = args.device
if args.classify_mass == 'True':
    label_names = ['Malign','Benign']
else:
    label_names = ['Mass']
class_size = len(label_names)
current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

results_dir = create_result_dir(current_dir)

config_paths, model_file_paths, model_result_paths, selected_model_names = get_choosen_models(current_dir, results_dir, model_names)

print(results_dir, 'directory created...')

if args.segment_breast =='True':
    seg_img_path = os.path.join(results_dir, 'breast_segmentation')
    print('Image files feeding to segmentation model and segmentation results will be saved on', seg_img_path, 'directory.'),
    crop_coordinates, img_shapes = apply_segmentation(seg_img_path, args.img_path, device)
    img_list, ann_list = control_annot_path(args.annotation_path, seg_img_path)
    if args.annotation_path:
        annot_path = get_annot_path(sorted(img_list), ann_list, img_shapes, crop_coordinates, seg_img_path)
        img_list, ann_list = control_annot_path(annot_path, seg_img_path)
    else:
        annot_path = args.annotation_path

else:
    crop_coordinates = None
    annot_path = args.annotation_path
    img_list, ann_list = control_annot_path(args.annotation_path, args.img_path)
    
if ann_list:
    bb_box, _ = get_gtbbox(ann_list)
    annot_classes = []
    for bb in bb_box:
        for b in bb:
            annot_classes.append(int(b[-1]))
    annot_classes = sorted(list(set(annot_classes)))
    if len(label_names) != len(annot_classes):
        print('According to class size, You must prepare annotation path or set --classu must give more than one model for applying ensemble stification parameter. You give'
             , len(annot_classes), 'classes in annotation file and you set --classification parameter as', args.classification, 'it causes not having same size of annotation and label classes. Label classes length and annotation classes length must be same.')
        parser.print_help()
        exit(-1)

write_labels_to_txt(label_names)

if args.coco_output_dir is None:
    coco_output_dir = os.path.join(results_dir, 'coco_predictions')
else:
    requested_coco_dir = args.coco_output_dir.strip()
    if requested_coco_dir:
        if os.path.isabs(requested_coco_dir):
            coco_output_dir = requested_coco_dir
        else:
            coco_output_dir = os.path.join(results_dir, requested_coco_dir)
    else:
        coco_output_dir = None

if coco_output_dir:
    os.makedirs(coco_output_dir, exist_ok=True)

coco_gt_path = args.gt_coco_path
coco_gt = None
image_id_map: Dict[str, int] = {}
category_ids: List[int] = []
enable_coco_artifacts = bool(coco_output_dir or coco_gt_path)

if enable_coco_artifacts:
    try:
        coco_gt, image_id_map, category_ids = _prepare_coco_resources(img_list, label_names, coco_gt_path)
    except (FileNotFoundError, KeyError) as coco_error:
        print('[WARNING] COCO export/evaluation disabled:', coco_error)
        coco_output_dir = None
        coco_gt = None
        image_id_map = {}
        category_ids = []
        enable_coco_artifacts = False

model_predicts = []
df_dict = {}
for i in range(len(config_paths)): 
    print('*'*20, selected_model_names[i], 'model evaluation processes are starting...', '*'*20)

    results = get_model_predicts(config_paths[i], model_file_paths[i], img_list, class_size, device)
    results = get_nms_results(results, img_list, class_size, args.nms_iou_threshold, scr_thr=args.confidence_threshold)
    if class_size == 2:
        results = filter_results(results)
    if ann_list:
        model_evals(config_paths[i], results, args.ap_threshold, args.img_path, annot_path, class_size)
    save_results(img_list, ann_list, results, label_names, model_result_paths[i])

    if enable_coco_artifacts and image_id_map:
        try:
            coco_predictions = _build_coco_predictions(img_list, results, image_id_map, category_ids)
        except KeyError as coco_error:
            print(f"[WARNING] Skipping COCO export for {selected_model_names[i]}:", coco_error)
        else:
            if coco_output_dir:
                output_file = os.path.join(
                    coco_output_dir,
                    f"{_slugify(selected_model_names[i])}_predictions.json",
                )
                with open(output_file, 'w', encoding='utf-8') as coco_file:
                    json.dump(coco_predictions, coco_file)
                print(f"COCO predictions saved to {output_file}")
            if coco_gt is not None:
                metrics = _compute_coco_metrics(coco_gt, coco_predictions, args.ap_threshold)
                print(
                    f"COCO metrics for {selected_model_names[i]} -> AP: {metrics['AP']:.4f}, "
                    f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
                )

    if args.enable_ensemble == 'True':
        model_predicts.append([results])

        
if len(model_predicts)!=0:
    print('*'*20, 'Applying Ensemble with:', ' '.join(selected_model_names), '*'*20)
    print('*'*20,'ENSEMBLE evaluation processes are starting...', '*'*20)
    
    ensemble_result_path = os.path.join(results_dir, '_'.join(model_names) + '_ensemble')
    print(ensemble_result_path)
    if not os.path.exists(ensemble_result_path):
        os.makedirs(ensemble_result_path)
    ensemble_result = applying_ensemble(img_list, model_predicts, class_size, args.nms_iou_threshold, args.confidence_threshold)
    if class_size == 2:
        ensemble_result = filter_results(ensemble_result)
    if ann_list:
        model_evals(config_paths[0], ensemble_result, args.ap_threshold, args.img_path, annot_path, class_size)
    save_results(img_list, ann_list, ensemble_result, label_names, ensemble_result_path)
    if enable_coco_artifacts and image_id_map:
        try:
            coco_predictions = _build_coco_predictions(img_list, ensemble_result, image_id_map, category_ids)
        except KeyError as coco_error:
            print('[WARNING] Skipping COCO export for ENSEMBLE:', coco_error)
        else:
            if coco_output_dir:
                output_file = os.path.join(
                    coco_output_dir,
                    f"{_slugify('ENSEMBLE')}_predictions.json",
                )
                with open(output_file, 'w', encoding='utf-8') as coco_file:
                    json.dump(coco_predictions, coco_file)
                print(f"COCO predictions saved to {output_file}")
            if coco_gt is not None:
                metrics = _compute_coco_metrics(coco_gt, coco_predictions, args.ap_threshold)
                print(
                    f"COCO metrics for ENSEMBLE -> AP: {metrics['AP']:.4f}, "
                    f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
                )
    selected_model_names.append('ENSEMBLE')

end_time = time.time()

print('-*- ELAPSED PROCESSING TIME:', int(end_time-start_time), 'seconds -*-')
