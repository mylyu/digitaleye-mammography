python infer_location_ensemble.py \
  --ensemble_dir ./out/learnable_fusion \
  --preds work_dirs/real_test/coco_predictions_corrected/atss_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/detr_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/faster_r-cnn_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/yolov3_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/ensemble_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/retinanet_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/varifocalnet_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/dynamic_r-cnn_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/deformable_detr_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/cascade_r-cnn_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/doublehead_r-cnn_predictions.json \
    work_dirs/real_test/coco_predictions_corrected/fcos_predictions.json \
  --out_json /mnt/d/BaiduNetdiskDownload/ISICDM2025/learnable_fusion_preds_submit.json \
  --category_mode pass_through\
  --nms_iou 0.05
