import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#     project="runs",
#     name="train/exp"
# )
#
# # Evaluate model performance on the validation set
# metrics = model.val(data='coco8.yaml', project="runs", name="val/exp")

# Perform object detection on an image
results = model("datasets/coco8/images/train/000000000009.jpg")
# res = results[0].plot()
boxes = results[0].boxes
a = boxes.cls.cpu().numpy()
res = results[0].plot()
print(results[0])
cv2.imshow("a", res)
# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭窗口
