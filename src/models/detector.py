from pathlib import Path

from ultralytics import YOLO

from src.config import MODELS_DIR


def resolve_detector_model_path(model_name=None):
    candidate_paths = []

    if model_name:
        candidate_paths.append(Path(model_name))

    candidate_paths.extend([
        MODELS_DIR / "detector_behavior_best.pt",
        MODELS_DIR / "checkpoints" / "yolo_objects_last.pt",
        Path("D:/EDITH/cognivision_yolo_outputs/best.pt"),
        Path("D:/EDITH/cognivision_yolo_outputs/last.pt"),
        Path("yolov8n.pt"),
    ])

    seen = set()
    for candidate in candidate_paths:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists():
            return candidate

    return Path(model_name) if model_name else Path("yolov8n.pt")

class CogniVisionDetector:
    """
    Handles multi-student and object detection using YOLOv8.
    Identifies 'person' and potentially 'cell phone' from COCO dataset.
    """
    def __init__(self, model_name=None, allowed_labels=None):
        resolved_model = resolve_detector_model_path(model_name)
        # Load the trained YOLO checkpoint when available, otherwise fall back safely.
        self.model = YOLO(str(resolved_model))
        self.classes = self.model.names
        self.allowed_labels = set(allowed_labels) if allowed_labels else None
        
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
            
            if self.allowed_labels is not None and label not in self.allowed_labels:
                continue

            if conf > 0.4:
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
