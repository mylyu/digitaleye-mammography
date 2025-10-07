# Digital Eye Mammography - Competition Ensemble

This repository reuses the pretrained detectors released in the original [Digital Eye for Mammography](https://github.com/ddobvyz/digitaleye-mammography) project and adds an anchor-wise, XGBoost-powered fusion layer designed for open mammography detection challenges. The convolutional backbones stay frozen: we rely on the published checkpoints to generate per-model predictions and learn a lightweight tree-based ensemble that improves leaderboard scores with minimal compute.

## Highlights
- **Pretrained foundations** - every detector checkpoint comes directly from the upstream Digital Eye release; no extra CNN training is required.
- **Learnable late fusion** - `train_location_ensemble.py` fits coordinate and IoU regressors (XGBoost by default, RandomForest fallback) on top of raw detector outputs.
- **Competition-friendly tooling** - scripts export COCO artefacts, evaluate splits, and package the ensemble for fast submission generation.
- **Reproducible and offline** - once the weights are downloaded, the full workflow runs without network access.

## Repository Layout
- `models/` - pretrained `.pth` weights mirrored from the upstream project.
- `configs/` - detector configuration files used by `mass_inference.py`.
- `mass_inference.py` - runs one or more detectors, optionally applies segmentation, and writes predictions (including COCO exports).
- `train_location_ensemble.py` - builds the XGBoost-based fusion model from training and validation predictions.
- `infer_location_ensemble.py` - loads the saved ensemble and produces competition-ready COCO JSON files.
- `utils/` - shared helpers for preprocessing, evaluation, and result export.

## Quick Start
1. **Create an environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download pretrained weights**
   - Grab the detector checkpoints from the original release (for example, `shared-models.v2` on GitHub).
   - Place each `.pth` file under `models/` so that the filename matches the associated config (e.g. `atss.pth` for `configs/0_atss_config.py`).

3. **Prepare your data**
   - `mass_inference.py` expects a directory of screening images (`--img_path`) and, optionally, a Digital Eye style annotation text file (`--annotation_path`) for evaluation.
   - If you supply `--gt_coco_path`, the script will compute COCO metrics and emit COCO-formatted predictions.
   - Images should already be converted to PNG/JPEG; the provided segmentation model can crop the breast region when `--segment_breast True`.

## Workflow

### 1. Generate detector predictions
Run the detectors you want to ensemble on both the training split (with labels) and the competition split (without labels). The command below saves raw results and COCO artefacts per model:

```bash
python mass_inference.py \
  --model_enum 0 1 2 6 8 \
  --img_path data/train/images \
  --annotation_path data/train/annotations.txt \
  --gt_coco_path data/train/gt_coco.json \
  --coco_output_dir out/train_coco \
  --device cuda:0 \
  --segment_breast True \
  --classify_mass False
```

Key tips:
- `--model_enum` accepts one or more detector ids (see the reference table below) and controls both the config and the checkpoint that are loaded.
- Results are written under `results/<detector_name>/`, and COCO predictions live in `results/<detector_name>/coco_predictions/` unless you override `--coco_output_dir`.
- Repeat the command for your validation/test splits so you have matching sets of JSON files for ensemble training and inference.

### 2. Train the XGBoost fusion
Supply aligned prediction JSON files for the training and validation (or public test) splits. The script builds anchor-wise features, fits the regressors, evaluates each detector, and writes the ensemble predictions for the validation split.

```bash
python train_location_ensemble.py \
  --train_gt data/train/gt_coco.json \
  --train_preds out/train_coco/atss_predictions.json out/train_coco/cascade_rcnn_predictions.json out/train_coco/deformable_detr_predictions.json \
  --test_gt data/val/gt_coco.json \
  --test_preds out/val_coco/atss_predictions.json out/val_coco/cascade_rcnn_predictions.json out/val_coco/deformable_detr_predictions.json \
  --outdir work_dirs/learnable_fusion \
  --label_iou_pos 0.4 \
  --min_neighbor_iou 0.05 \
  --coord_reg xgb \
  --iou_reg xgb
```

What you get:
- `work_dirs/learnable_fusion/coord_reg.joblib` and `iou_reg.joblib` - the fitted ensemble.
- `work_dirs/learnable_fusion/learnable_fusion_preds.json` - ensemble predictions on the validation split.
- Console metrics comparing single-model baselines and the fused result at multiple IoU thresholds.

### 3. Produce competition submissions
Load the saved regressors and pass the detector predictions for the unlabeled competition set. Control how categories are assigned using `--category_mode`.

```bash
python infer_location_ensemble.py \
  --ensemble_dir work_dirs/learnable_fusion \
  --preds out/test_coco/atss_predictions.json out/test_coco/cascade_rcnn_predictions.json out/test_coco/deformable_detr_predictions.json \
  --images_json data/test/images.json \
  --out_json submissions/ensemble_predictions.json \
  --category_mode fixed \
  --fixed_category_id 1 \
  --nms_iou 0.5
```

The script constructs a full COCO dictionary (reusing metadata if available), applies non-maximum suppression, and ensures submission-ready formatting.

## Model Enum Reference

| Enum | Detector        | Checkpoint          |
|------|-----------------|---------------------|
| 0    | ATSS            | `models/atss.pth` |
| 1    | Cascade R-CNN   | `models/cascade_rcnn.pth` |
| 2    | Deformable DETR | `models/deformable_detr.pth` |
| 3    | DETR            | `models/detr.pth` |
| 4    | Double-Head R-CNN | `models/doublehead_rcnn.pth` |
| 5    | Dynamic R-CNN   | `models/dynamic_rcnn.pth` |
| 6    | Faster R-CNN    | `models/fasterrcnn.pth` |
| 7    | FCOS            | `models/fcos.pth` |
| 8    | RetinaNet       | `models/retina_net.pth` |
| 9    | VarifocalNet    | `models/varifocal_net.pth` |
| 10   | YOLOv3          | `models/yolo_v3.pth` |

Keep the model order consistent between `--train_preds`, `--test_preds`, and `--preds`, as the ensemble relies on aligned feature layouts.

## Evaluation and Utilities
- `merge_coco_gt_and_predictions.py` can merge ensemble outputs with ground truth for local analysis.
- `infer_location_ensemble.sh` and `train_location_ensemble.sh` are convenience wrappers for common experiment setups.
- Set `--anchor_score_thresh` in the ensemble scripts to filter weak anchors before feature construction.

## Notes on Data Privacy and Compliance
The upstream KETEM dataset is not included. Ensure you have rights to any screening data you process, de-identify sensitive information, and follow competition rules when sharing predictions or models.

## Acknowledgements
This work is built on the Digital Eye project by the Digital Transformation Office of the Presidency of the Republic of Türkiye. All pretrained checkpoints are sourced from their public releases; please cite the original paper when appropriate.

## License
This repository remains under the terms of the [GNU GPLv3](LICENSE).

## Disclaimer
The code and models are supplied "as is", without any warranty of fitness for medical diagnosis or clinical decision-making. Always involve qualified healthcare professionals when interpreting mammography findings.
