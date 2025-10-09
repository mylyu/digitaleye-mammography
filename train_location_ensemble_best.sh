python train_location_ensemble3.py \
  --train_gt annotations/instances_train_final.json \
  --train_preds work_dirs/train/coco_predictions_corrected/atss_predictions.json \
    work_dirs/train/coco_predictions_corrected/detr_predictions.json \
    work_dirs/train/coco_predictions_corrected/faster_r-cnn_predictions.json \
    work_dirs/train/coco_predictions_corrected/yolov3_predictions.json \
    work_dirs/train/coco_predictions_corrected/ensemble_predictions.json \
    work_dirs/train/coco_predictions_corrected/retinanet_predictions.json \
    work_dirs/train/coco_predictions_corrected/varifocalnet_predictions.json \
    work_dirs/train/coco_predictions_corrected/dynamic_r-cnn_predictions.json \
    work_dirs/train/coco_predictions_corrected/deformable_detr_predictions.json \
    work_dirs/train/coco_predictions_corrected/cascade_r-cnn_predictions.json \
    work_dirs/train/coco_predictions_corrected/doublehead_r-cnn_predictions.json \
    work_dirs/train/coco_predictions_corrected/fcos_predictions.json \
  --test_gt annotations/instances_test_final.json \
  --test_preds work_dirs/test/coco_predictions_corrected/atss_predictions.json \
    work_dirs/test/coco_predictions_corrected/detr_predictions.json \
    work_dirs/test/coco_predictions_corrected/faster_r-cnn_predictions.json \
    work_dirs/test/coco_predictions_corrected/yolov3_predictions.json \
    work_dirs/test/coco_predictions_corrected/ensemble_predictions.json \
    work_dirs/test/coco_predictions_corrected/retinanet_predictions.json \
    work_dirs/test/coco_predictions_corrected/varifocalnet_predictions.json \
    work_dirs/test/coco_predictions_corrected/dynamic_r-cnn_predictions.json \
    work_dirs/test/coco_predictions_corrected/deformable_detr_predictions.json \
    work_dirs/test/coco_predictions_corrected/cascade_r-cnn_predictions.json \
    work_dirs/test/coco_predictions_corrected/doublehead_r-cnn_predictions.json \
    work_dirs/test/coco_predictions_corrected/fcos_predictions.json \
  --outdir ./out/learnable_fusion_best \
  --label_iou_pos 0.075 \
  --min_neighbor_iou 0.02 \
  --anchor_score_thresh 0.00 \
  --nms_iou 0.60 \
  --coord_reg xgb \
  --iou_reg xgb \
  --cls_model xgb
