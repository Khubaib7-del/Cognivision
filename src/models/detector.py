from ultralytics import YOLO
import cv2
import torch

class CogniVisionDetector:
    """
    Handles multi-student and object detection using YOLOv8.
    Identifies 'person' and potentially 'cell phone' from COCO dataset.
    """
    def __init__(self, model_name='yolov8n.pt'):
        # Load the YOLOv8 task (nano version for speed)
        self.model = YOLO(model_name)
        self.classes = self.model.names
        
    def detect_students(self, frame):
        """
        Runs inference on a frame and returns bounding boxes for students.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.classes[cls_id]
            conf = float(box.conf[0])
            
            # We care about 'person' (ID 0) and 'cell phone' (ID 67)
            if label in ['person', 'cell phone'] and conf > 0.4:
                coords = box.xyxy[0].tolist() # x1, y1, x2, y2
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': [int(c) for c in coords]
                })
                
        return detections

if __name__ == "__main__":
    # Smoke test
    detector = CogniVisionDetector()
    print("YOLOv8 Detector initialized successfully.")
    print(f"Classes available: {len(detector.classes)}")
