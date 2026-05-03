

"""webcam_opencv_dnn_yolo_coco.py: test of opencv yolo with a webcam"""

__author__      = "Fabrice Jumel"
__license__ = "CC0"
__version__ = "0.1"

import cv2
import numpy as np


net = cv2.dnn.readNet("detection/yolov3-tiny.weights", "detection/yolov3-tiny.cfg")


with open("detection/coco.names", "r") as f:
    classes = f.read().strip().split("\n")


cap = cv2.VideoCapture(0)  #il faut changer selon votre ordinateur
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);

while True:
    ret, frame = cap.read()  

    if not ret:
        break


    height, width = frame.shape[:2]


    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)


    net.setInput(blob)


    layer_names = net.getUnconnectedOutLayersNames()

    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    conf_threshold = 0.5

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

