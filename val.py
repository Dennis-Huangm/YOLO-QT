# Denis
# -*- coding: utf-8 -*-
import os

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load a model
model = YOLO("yolov8n.pt")

# Customize validation settings
# validation_results = model.val(data="coco8.yaml", imgsz=640, batch=2, conf=0.25, iou=0.6, device="0")
metrics = model.val(data="coco8.yaml", project="runs", name="val/exp")
