import os
import cv2
import numpy as np
import onnxruntime as ort


class MaskedFaceDetector:
    def __init__(self, label_path, onnx_path, prob_threshold=0.7, iou_threshold=0.7, top_k=-1):
        self.class_names = [name.strip() for name in open(label_path).readlines()]
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.top_k = top_k

    def predict(self, image):
        height, width, _ = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, labels, probs = self._post_process(width, height, confidences, boxes)

        return boxes, labels, probs

    def _post_process(self, width, height, confidences, boxes):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []

        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > self.prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self._hard_nms(box_probs)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])

        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])

        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height

        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def _hard_nms(self, box_scores):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-self.top_k:]

        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)

            if 0 < self.top_k == len(picked) or len(indexes) == 1:
                break

            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self._iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
            indexes = indexes[iou <= self.iou_threshold]

        return box_scores[picked, :]

    @staticmethod
    def _iou_of(boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = MaskedFaceDetector._area_of(overlap_left_top, overlap_right_bottom)
        area0 = MaskedFaceDetector._area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = MaskedFaceDetector._area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    @staticmethod
    def _area_of(left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]


