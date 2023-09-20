from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')
pic=cv2.imread(r"E:/pycharm/pythonProject/ObjectDetection/images/School.png")
result = model(pic)
cv2.imshow("output",pic)