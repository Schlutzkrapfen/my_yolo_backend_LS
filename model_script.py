
from pathlib import Path

from label_studio_ml.model import LabelStudioMLBase
from torch import os
from ultralytics import YOLO
import cv2

LOCAL_FOLDER:Path = Path("~/myfiles")
def make_local_url(image_path):
    parts = image_path.split("/")
    local_path =  LOCAL_FOLDER.joinpath(*parts[4:])
    local_path = os.path.expanduser(local_path)
    return local_path



class YOLOTopKBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        """
        initializes the YOLOTopKBackend with the given keyword arguments
        """
        super().__init__(**kwargs)
        self.model = YOLO("caput.pt")
        self.cap_class_id = 0
        self.K = 2

    def predict(self, tasks, **kwargs):
        """
        predicts the top K bounding boxes for each task
        """
        predictions = []

        for task in tasks:
            image_path = task["data"]["image"]  # Pfad/URL aus Label Studio
            image_path = make_local_url(image_path)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                continue

            img_h, img_w = img.shape[:2]

            results = self.model.predict(source=image_path, conf=0.05, iou=0.5)
            boxes = results[0].boxes
            if not boxes:
                print(f"No boxes found in image: {image_path}")
                continue

            mask = boxes.cls == self.cap_class_id
            cap_boxes = boxes[mask]
            sorted_idx = (-cap_boxes.conf).argsort()
            top_k_boxes = cap_boxes[sorted_idx][: self.K]

            result_items = []
            for box in top_k_boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # Label Studio erwartet % statt normalisierter Werte
                result_items.append({
                    "from_name": "TMJ",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x1 / img_w * 100,
                        "y": y1 / img_h * 100,
                        "width": (x2 - x1) / img_w * 100,
                        "height": (y2 - y1) / img_h * 100,
                        "rectanglelabels": ["Asymmetry Caput"],
                    },
                    "score": conf,
                })

            predictions.append({
                "result": result_items,
                "score": sum(item["score"] for item in result_items) / len(result_items) if result_items else 0,
            })
        if not predictions:
            print("something went wrong not sure where")
            return None

        return predictions
