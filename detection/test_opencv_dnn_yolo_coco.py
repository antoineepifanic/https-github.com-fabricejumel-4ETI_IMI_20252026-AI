import cv2
import numpy as np
import argparse
import json
import os
import math


# Q2.1Configuration via ligne de commande

parser = argparse.ArgumentParser(description='Détection YOLO avec filtres, JSON et Mosaïque')
parser.add_argument('--input', type=str, default='0', help='Chemin vers une image, une vidéo, ou "0" pour la webcam')
parser.add_argument('--classes', type=str, nargs='+', default=[], help='Liste des classes à filtrer (ex: person dog car). Vide = tout afficher.')
parser.add_argument('--output_dir', type=str, default='output', help='Dossier où sauvegarder les crops et le JSON')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

net = cv2.dnn.readNet("detection/yolov3-tiny.weights", "detection/yolov3-tiny.cfg")
with open("detection/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

is_image = args.input.lower().endswith(('.png', '.jpg', '.jpeg'))
if is_image:
    cap = None
    frame = cv2.imread(args.input)
else:
    source = int(args.input) if args.input == '0' else args.input
    cap = cv2.VideoCapture(source)

json_data = {
    "source": args.input,
    "detections": []
}
crops_by_class = {c: [] for c in classes} 
frame_count = 0

while True:
    if not is_image:
        ret, frame = cap.read()
        if not ret:
            break
    
    frame_count += 1
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    class_ids, confidences, boxes = [], [], []
    conf_threshold = 0.2

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            label = str(classes[class_ids[i]])
            
            # Q2.1Filtrage des classes
            if len(args.classes) > 0 and label not in args.classes:
                continue

            confidence = confidences[i]
            x, y, w, h = boxes[i]
            
            x_start, y_start = max(0, x), max(0, y)
            x_end, y_end = min(width, x + w), min(height, y + h)

            # Q2.2Stockage pour le JSON
            json_data["detections"].append({
                "frame": frame_count,
                "class": label,
                "confidence": round(confidence, 2),
                "bounding_box": {"x": x_start, "y": y_start, "w": w, "h": h}
            })

            # Q2.3Sauvegarde de la Bounding Box sous forme d'image
            if y_end > y_start and x_end > x_start:
                crop_img = frame[y_start:y_end, x_start:x_end]
                crops_by_class[label].append(crop_img)
                
                crop_filename = f"{args.output_dir}/{label}_f{frame_count}_{i}.jpg"
                cv2.imwrite(crop_filename, crop_img)

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x_start, y_start - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection YOLO", frame)


    if is_image:
        cv2.waitKey(0)
        break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if not is_image:
    cap.release()
cv2.destroyAllWindows()


# SAUVEGARDE FINALE (JSON & MOSAÏQUE)

json_path = os.path.join(args.output_dir, "detections.json")
with open(json_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)
print(f"[*] Données JSON sauvegardées dans {json_path}")


print("[*] Génération des mosaïques...")
for label, crops in crops_by_class.items():
    if len(crops) > 0:
        size = 100
        resized_crops = [cv2.resize(img, (size, size)) for img in crops]
        
        grid_size = math.ceil(math.sqrt(len(resized_crops)))
        
        mosaic = np.zeros((grid_size * size, grid_size * size, 3), dtype=np.uint8)
        
        for idx, crop in enumerate(resized_crops):
            row = idx // grid_size
            col = idx % grid_size
            mosaic[row*size:(row+1)*size, col*size:(col+1)*size] = crop
            
        mosaic_path = os.path.join(args.output_dir, f"mosaic_{label}.jpg")
        cv2.imwrite(mosaic_path, mosaic)
        print(f"    -> Mosaïque créée pour '{label}': {mosaic_path}")

print("[*] Terminé !")