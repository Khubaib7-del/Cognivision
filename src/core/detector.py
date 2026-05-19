from ultralytics import YOLO
from pathlib import Path


class CogniVisionDetector:
    """
    Simple unified detector wrapper for YOLOv8.
    
    Supports both standard models (yolov8n.pt, yolov8m.pt) and custom trained models.
    No filtering or class restrictions - returns all detections.
    """
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = "yolov8n.pt"
        
        # Try to resolve path for custom models
        model_path = self._resolve_model_path(model_name)
        
        # Load YOLO model (auto-downloads if standard model)
        try:
            self.model = YOLO(str(model_path))
            print(f"✓ Loaded detector: {model_path}")
        except Exception as e:  # noqa: Broad exception catch for model loading fallback
            print(f"Warning: Failed to load {model_path}: {e}")
            print("Falling back to yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
        
        self.classes = self.model.names
    
    def _resolve_model_path(self, model_name):
        """Resolve model path for both standard and custom models."""
        # If it's a standard model name, return as-is
        if isinstance(model_name, str) and model_name.startswith("yolov8") and model_name.endswith(".pt"):
            return model_name
        
        # Convert to Path for better handling
        model_path = Path(model_name)
        
        # If it's an absolute path that exists, use it directly
        if model_path.exists():
            return model_path
        
        # Try to find custom model in known locations
        from src.config import MODELS_DIR
        
        candidates = [
            MODELS_DIR / model_name,  # Local models dir
            Path(r"D:\EDITH\cognivision_yolo_outputs") / model_name,  # YOLO outputs root
            Path(r"D:\EDITH\cognivision_yolo_outputs\cognivision_yolo\objects_v1\weights") / model_name,  # Track 1
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # If nothing found, return original (will try to download or error)
        return model_path
        
    def detect_students(self, frame):
        """
        Run inference on a frame and return all detections.
        
        Returns list of dicts with:
            - label: class name
            - confidence: detection confidence
            - bbox: [x1, y1, x2, y2] coordinates
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.classes[cls_id]
            conf = float(box.conf[0])
            
            # Keep all detections with confidence > threshold
            if conf > 0.25:
                coords = box.xyxy[0].tolist()  # x1, y1, x2, y2
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

