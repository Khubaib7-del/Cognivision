import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

from src.core.detector import CogniVisionDetector
from src.core.efficientnet_classifier import load_efficientnet_model
from src.config import CLASSIFIER_INPUT_SIZE, DEVICE, MODELS_DIR

class ProcessingPipeline:
    """
    Unified pipeline combining YOLOv8 detection + EfficientNet-B0 attention classification.
    """
    def __init__(self, detector_model='yolov8n.pt', classifier_weights=None):
        # Initialize Detector
        self.detector = CogniVisionDetector(model_name=detector_model)
        
        # Initialize Classifier (EfficientNet-B0 trained on FER2013)
        if classifier_weights is None:
            classifier_weights = MODELS_DIR / "efficientnet_attention_best.pth"
        
        if isinstance(classifier_weights, str):
            classifier_weights = Path(classifier_weights)
        
        self.classifier = load_efficientnet_model(
            path=str(classifier_weights) if classifier_weights.exists() else None,
            device=DEVICE
        )
        
        # Setup transforms (224x224 for EfficientNet)
        self.transform = transforms.Compose([
            transforms.Resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.labels = ["distracted", "attentive"]

    def _extract_face_region(self, frame, bbox, face_height_ratio=0.45):
        """
        Extract face region from full-body bounding box.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            face_height_ratio: Portion of height to use for face (top portion has eyes/mouth)
        
        Returns:
            Face crop or None if invalid
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Extract top portion for face (eyes + mouth are in upper region)
        height = y2 - y1
        face_height = int(height * face_height_ratio)
        face_y2 = y1 + face_height
        
        face_crop = frame[y1:face_y2, x1:x2]
        
        # Validate crop size
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return None
        
        return face_crop

    def process_frame(self, frame, return_crops=False):
        """
        Process a single frame: detect persons, classify attention.
        Integrates YOLO detection + EfficientNet classification.
        
        Returns:
            List of detections with classification results and confidence scores.
            If return_crops=True, returns (results, face_crops) tuple.
        """
        detections = self.detector.detect_students(frame)
        
        results = []
        crops = []
        
        for det in detections:
            label = det['label']
            bbox = det['bbox']
            conf = det['confidence']
            
            if label == 'person':
                # Extract face region from detection bbox
                face_crop = self._extract_face_region(frame, bbox, face_height_ratio=0.45)
                
                if face_crop is not None:
                    # Classify attention using EfficientNet
                    try:
                        # Convert BGR -> RGB and apply transforms
                        pil_image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        tensor_image = self.transform(pil_image).unsqueeze(0).to(DEVICE)
                        
                        # Run inference
                        with torch.no_grad():
                            output = self.classifier(tensor_image)
                            probabilities = torch.softmax(output, dim=1)
                            predicted_class = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0, predicted_class].item()
                        
                        status = self.labels[predicted_class]
                        
                        results.append({
                            'type': 'student',
                            'bbox': bbox,
                            'status': status,
                            'confidence': confidence,
                            'detection_confidence': conf
                        })
                        
                        if return_crops:
                            crops.append(face_crop)
                    
                    except Exception as e:
                        print(f"⚠ Classification error for person at {bbox}: {e}")
                        results.append({
                            'type': 'student',
                            'bbox': bbox,
                            'status': 'unknown',
                            'confidence': 0.0,
                            'detection_confidence': conf,
                            'error': str(e)
                        })
                else:
                    # Face crop extraction failed (too small or invalid)
                    results.append({
                        'type': 'student',
                        'bbox': bbox,
                        'status': 'unknown',
                        'confidence': 0.0,
                        'detection_confidence': conf,
                        'error': 'face_crop_failed'
                    })
            
            elif label == 'cell phone':
                # Phone detection - flagged as distraction
                results.append({
                    'type': 'distraction',
                    'bbox': bbox,
                    'status': 'phone_detected',
                    'confidence': conf,
                    'detection_confidence': conf
                })
        
        if return_crops:
            return results, crops
        return results
