import argparse
import cv2
import numpy as np


CLASSES = ['doughnut', 'pizza', 'spaghetti', 'citrouille']  #classes faites dans le modèle


colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Dessine les boîtes englobantes sur l'image.
    """
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    """
    Charge le modèle ONNX, effectue l'inférence et affiche les détections.
    """
    #Chargement du modèle ONNX
    model = cv2.dnn.readNetFromONNX(onnx_model)

    #lecture de l'image
    original_image = cv2.imread(input_image)
    if original_image is None:
        print(f"[ERREUR] Impossible de charger l'image : {input_image}")
        return
    height, width = original_image.shape[:2]


    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image


    INPUT_SIZE = 640  


    scale = length / INPUT_SIZE  

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255,
                                  size=(INPUT_SIZE, INPUT_SIZE), swapRB=True)
    model.setInput(blob)

  
    outputs = model.forward()

   
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Parcours des prédictions
    for i in range(rows):
        class_scores = outputs[0][i][4:]
        max_score = float(np.max(class_scores))
        max_class_id = int(np.argmax(class_scores))

  
        if max_score >= 0.45: #seil de confiance

            cx, cy, w, h = outputs[0][i][:4]
            box = [
                (cx - 0.5 * w) * scale,
                (cy - 0.5 * h) * scale,
                w * scale,
                h * scale
            ]
            boxes.append(box)
            scores.append(max_score)
            class_ids.append(max_class_id)


    indices = cv2.dnn.NMSBoxes(
        boxes, scores,
        score_threshold=0.45,
        nms_threshold=0.5
    )


    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            draw_bounding_box(original_image, class_ids[i], scores[i],
                              x, y, x + w, y + h)

    #affichage
    cv2.imshow('Détection ONNX', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='mon_modele.onnx',
                        help='Chemin vers votre modèle ONNX.')

    parser.add_argument('--img', default='test.jpg',
                        help='Chemin vers l\'image d\'entrée.')
    args = parser.parse_args()
    main(args.model, args.img)