import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.detector import CogniVisionDetector
from models.attention_classifier import AttentionClassifier

class CogniVisionEngine:
    """
    The central coordinator for CogniVision's AI pipeline.
    Combines Detection (YOLOv8) with Classification (AttentionClassifier).
    """
    def __init__(self, classifier_weights="src/models/attention_model.pth"):
        # 1. Initialize Detector
        self.detector = CogniVisionDetector()
        
        # 2. Initialize Classifier
        self.classifier = AttentionClassifier()
        if torch.cuda.is_available():
            self.classifier.load_state_dict(torch.load(classifier_weights))
            self.classifier.to("cuda")
        else:
            self.classifier.load_state_dict(torch.load(classifier_weights, map_location="cpu"))
        self.classifier.eval()
        
        # 3. Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.labels = ["attentive", "distracted"]

    def process_frame(self, frame):
        """
        Runs the full pipeline on a single frame.
        """
        # A. Detect people and phones
        detections = self.detector.detect_students(frame)
        
        results = []
        for det in detections:
            label = det['label']
            bbox = det['bbox'] # [x1, y1, x2, y2]
            
            if label == 'person':
                # B. Crop face area (simple heuristic: top 1/3 of bounding box)
                x1, y1, x2, y2 = bbox
                face_y2 = y1 + (y2 - y1) // 3
                face_crop = frame[y1:face_y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                # C. Classify attention
                attention_status = self._classify_attention(face_crop)
                results.append({
                    'type': 'student',
                    'status': attention_status,
                    'bbox': bbox
                })
            else:
                # D. Just record phone as a separate entity
                results.append({
                    'type': 'object',
                    'status': 'distraction (phone)',
                    'bbox': bbox
                })
                
        return results

    def _classify_attention(self, crop):
        """
        Helper to run the classifier on a cropped image.
        """
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.to("cuda")
            
        with torch.no_grad():
            output = self.classifier(img_tensor)
            _, predicted = torch.max(output, 1)
            
        return self.labels[predicted.item()]

if __name__ == "__main__":
    # Smoke test
    try:
        engine = CogniVisionEngine()
        print("CogniVision Engine initialized successfully.")
    except Exception as e:
        print(f"Engine initialization failed: {e}")
