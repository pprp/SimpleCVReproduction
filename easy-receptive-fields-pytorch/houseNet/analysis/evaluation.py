import json

import cv2
import os
import keras
import numpy as np

from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from util.array_utils import image_as_array


class Evaluator:
    """Pipeline for evaluating different metrics.

    # Arguments:
        data_path: str
            absolute path to validation directory
        model: keras.Model
            trained model
    """

    def __init__(self: 'Evaluator',
                 data_path: str,
                 model: keras.Model):
        self.images_path = os.path.join(data_path, 'images')
        self.annotations_path = os.path.join(data_path, 'annotation-small.json')
        self.coco = COCO(self.annotations_path)
        self.model = model

    def evaluate(self, predictions):
        """Performs the evaluation.

        # Arguments:
            predictions: [obj]
                list of coco annotated predictions
        """
        results = self.coco.loadRes(predictions)
        coco_eval = COCOeval(self.coco, results, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def save_predictions_as_json(self, prediction_file_name: str, batch_size: int = 8):
        images_ids = self.coco.getImgIds()
        n_batches = int(np.ceil(len(images_ids) / batch_size))

        annotations = []
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(images_ids))
            batch_ids = images_ids[batch_start:batch_end]

            images_annotations = self.coco.loadImgs(batch_ids)
            images = np.array([image_as_array(self.images_path, i) for i in images_annotations])

            predictions = self.model.predict(images / 255, batch_size=batch_size)
            predictions_annotation = self._masks_to_annotations(batch_ids, predictions)

            annotations.extend(predictions_annotation)
            print("Finished %.2f %%" % (100 * i / n_batches))
        self._save_all_annotations(prediction_file_name, annotations)

    def _masks_to_annotations(self, image_ids, masks):
        annotations = []
        for ID, m in zip(image_ids, masks):
            m = m.astype(np.uint8)
            m = np.squeeze(m)
            n, labels = cv2.connectedComponents(m)
            for i in range(1, n + 1):
                label = (labels == i).astype(np.uint8)
                annotations.append(self._mask_to_annotation(ID, label))

        return annotations

    def _mask_to_annotation(self, image_id, label):
        label = np.asfortranarray(label, dtype=np.uint8)
        rle = mask.encode(label)
        rle['counts'] = rle['counts'].decode('UTF-8')
        return {
            'image_id': image_id,
            'category_id': 100,
            'score': 1,
            'segmentation': rle,
            'bbox': mask.toBbox(rle).tolist()
        }

    def _save_all_annotations(self, prediction_file_path, annotations):
        path, _ = os.path.split(prediction_file_path)
        if not os.path.exists(path):
            os.mkdir(path)
        with open(prediction_file_path, 'w+') as prediction_file:
            prediction_file.truncate(0)
            prediction_file.write(json.dumps(annotations))
