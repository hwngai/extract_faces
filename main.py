from detect_face import MaskedFaceDetector
import cv2
import os

label_path = "models/voc-model-labels.txt"
onnx_path = "models/RFB-320-face-v2.onnx"
faceDetector = MaskedFaceDetector(label_path, onnx_path)

input_folder = "input_images"
output_folder = "face_images"


def calculate_face_area_percentage(img_width, img_height, boxes_face):
    img_area = img_width * img_height

    total_face_area = 0
    for box in boxes_face:
        x, y, w, h = box
        face_area = w * h
        total_face_area += face_area

    face_area_percentage = (total_face_area / img_area) * 100

    return face_area_percentage

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        file_path = os.path.join(input_folder, filename)
        img = cv2.imread(file_path)
        boxes_face, labels_face, probs_face = faceDetector.predict(img)
        if boxes_face.shape[0] == 1:
            img_height, img_width, _ = img.shape
            face_area_percentage = calculate_face_area_percentage(img_width, img_height, boxes_face)
            if face_area_percentage > 60:
                print(boxes_face, labels_face, probs_face)
                _, file_name = os.path.split(file_path)
                output_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_path, img)









