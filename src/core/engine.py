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
        import os
        if os.path.exists(classifier_weights):
            try:
                if torch.cuda.is_available():
                    self.classifier.load_state_dict(torch.load(classifier_weights))
                    self.classifier.to("cuda")
                else:
                    self.classifier.load_state_dict(torch.load(classifier_weights, map_location="cpu"))
            except Exception as e:
                print(f"Phase 1 Warning: Old model weights incompatible. Did you train the new Custom CNN? Error: {e}")
        else:
            print("Phase 1 Warning: attention_model.pth not found. The model will use random initialization. Please train it first.")
        
        self.classifier.eval()
        
        # 3. Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.labels = ["attentive", "distracted"]

    def process_frame(self, frame, return_crops=False):
        """
        Runs the full pipeline on a single frame.
        """
        detections = self.detector.detect_students(frame)
        
        results = []
        crops = []
        for det in detections:
            label = det['label']
            bbox = det['bbox']
            
            if label == 'person':
                # B. Crop face area (Improved: top 45% for better eye/face coverage)
                x1, y1, x2, y2 = bbox
                h = y2 - y1
                face_y2 = y1 + int(h * 0.45) 
                face_crop = frame[y1:face_y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                # C. Classify attention
                attention_status, confidence = self._classify_attention(face_crop)
                results.append({
                    'type': 'student',
                    'status': attention_status,
                    'confidence': confidence,
                    'bbox': bbox
                })
                if return_crops:
                    crops.append(face_crop)
            else:
                results.append({
                    'type': 'object',
                    'status': 'distraction (phone)',
                    'confidence': det['confidence'],
                    'bbox': bbox
                })
                
        if return_crops:
            return results, crops
        return results

    def _classify_attention(self, crop):
        """
        Helper to run the classifier on a cropped image.
        Returns (label, confidence).
        """
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.to("cuda")
            
        with torch.no_grad():
            output = self.classifier(img_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)
            
        return self.labels[predicted.item()], confidence.item()

if __name__ == "__main__":
    # Smoke test
    try:
        engine = CogniVisionEngine()
        print("CogniVision Engine initialized successfully.")
    except Exception as e:
        print(f"Engine initialization failed: {e}")
