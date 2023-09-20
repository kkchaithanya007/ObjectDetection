import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov5s.pt')

# Perform object detection on an image
results = model('E:\pycharm\pythonProject\ObjectDetection\images\School.png')  # Replace 'path_to_image.jpg' with the path to your image file

# Load the image using OpenCV
image = cv2.imread('E:\pycharm\pythonProject\ObjectDetection\images\School.png')  # Replace 'path_to_image.jpg' with the path to your image file

# Loop through the detected objects and draw bounding boxes
for det in results.xyxy[0]:
    conf = det[4].item()
    label = int(det[5].item())
    class_name = results.names[0][label]
    box = det[0:4].int().tolist()

    if conf > 0.5:  # You can adjust the confidence threshold as needed
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {conf:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
