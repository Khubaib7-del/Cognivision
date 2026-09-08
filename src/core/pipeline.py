import cv2
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from src.core.detector import CogniVisionDetector
from src.core.efficientnet_classifier import load_efficientnet_model, heuristic_classification
from src.config import (
    CLASSIFIER_INPUT_SIZE, DEVICE, MODELS_DIR, DETECTOR_MODEL, 
    BEHAVIOR_DETECTOR_MODEL, CLASSIFIER_MODEL, CLASSIFIER_LABELS,
    STUDENT_LABELS
)

class ProcessingPipeline:
    """
    RESTRUCTURED: Dual detector with clean categorization (NO suppression).
    - Detector 1 (yolov8n.pt): person + generic objects from COCO
    - Detector 2 (Track 1 best.pt): cheating behaviors including "using mobile"
    - EfficientNet classification for student attention
    - All detections returned together - NO suppression logic
    """
    def __init__(self, classifier_weights=None):
        # Detector 1: Custom trained YOLO model for detection
        self.detector_coco = CogniVisionDetector(model_name=str(DETECTOR_MODEL))
        
        # Detector 2: Cheating behaviors (custom trained model) - OPTIONAL if model doesn't exist
        # Try to load behavior model, fallback gracefully
        try:
            self.detector_behavior = CogniVisionDetector(model_name=str(BEHAVIOR_DETECTOR_MODEL))
            self.has_behavior_detector = True
        except:
            print("⚠ Behavior detector not found. Using COCO only.")
            self.detector_behavior = None
            self.has_behavior_detector = False

        # Lightweight face detector used to localize the actual face inside a person box.
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier"):
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        # Initialize Classifier (EfficientNet RGB checkpoint from Kaggle training)
        if classifier_weights is None:
            classifier_weights = CLASSIFIER_MODEL
        
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
        
        self.labels = CLASSIFIER_LABELS
        self.student_labels = set(STUDENT_LABELS)
        
        # Define object categories for clean output organization
        self.phone_labels = {"cell phone", "phone", "mobile", "handbag"}
        self.cheating_labels = {"leaning to copy", "looking around", "sharing_answers", "using mobile", "using_mobile"}

    def _extract_face_region(self, frame, bbox, face_height_ratio=0.35, face_width_ratio=0.7):
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
        
        # Extract a tighter face-focused crop from the upper body box.
        height = y2 - y1
        face_height = int(height * face_height_ratio)
        face_y2 = y1 + face_height

        width = x2 - x1
        face_width = int(width * face_width_ratio)
        face_x1 = x1 + (width - face_width) // 2
        face_x2 = face_x1 + face_width

        face_crop = frame[y1:face_y2, face_x1:face_x2]
        
        # Validate crop size
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return None
        
        return face_crop

    def _get_face_bbox(self, frame, bbox, face_height_ratio=0.35, face_width_ratio=0.7):
        """Return the upper face region bounding box derived from a person box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        height = y2 - y1
        face_height = max(1, int(height * face_height_ratio))
        face_y2 = min(h, y1 + face_height)
        width = x2 - x1
        face_width = max(1, int(width * face_width_ratio))
        face_x1 = max(0, x1 + (width - face_width) // 2)
        face_x2 = min(w, face_x1 + face_width)

        return [face_x1, y1, face_x2, face_y2]

    def _detect_face_bbox(self, frame, bbox):
        """Detect the most likely face inside a detected person box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        person_crop = frame[y1:y2, x1:x2]
        if self.face_cascade is None:
            return self._get_face_bbox(frame, bbox)

        if person_crop.size == 0:
            return self._get_face_bbox(frame, bbox)

        gray = (0.114 * person_crop[:, :, 0] + 0.587 * person_crop[:, :, 1] + 0.299 * person_crop[:, :, 2]).astype('uint8')

        # Focus the detector on the upper portion of the person box where the face should be.
        upper_limit = max(1, int(gray.shape[0] * 0.75))
        upper_gray = gray[:upper_limit, :]

        faces = self.face_cascade.detectMultiScale(
            upper_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = [int(v) for v in face]
            return [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]

        return self._get_face_bbox(frame, bbox)

    def _bbox_iou(self, bbox_a, bbox_b):
        """Return IoU between two [x1, y1, x2, y2] boxes."""
        ax1, ay1, ax2, ay2 = [int(v) for v in bbox_a]
        bx1, by1, bx2, by2 = [int(v) for v in bbox_b]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def process_frame(self, frame, return_crops=False):
        """
        RESTRUCTURED: Dual-detector inference with clean categorization.
        
        - Run both COCO and behavior detectors
        - Categorize outputs: student (person), phone (using mobile), objects
        - Classify student attention with EfficientNet
        - NO suppression logic - all boxes returned together
        
        Returns:
            List of detections with clean categorization.
            If return_crops=True, returns (results, face_crops) tuple.
        """
        # Inference Pass 1: Generic objects (COCO - person, etc.)
        coco_detections = self.detector_coco.detect_students(frame)
        
        # Inference Pass 2: Behavior detections (Track 1 - using mobile, etc.)
        behavior_detections = [] if not self.has_behavior_detector else self.detector_behavior.detect_students(frame)
        
        results = []
        crops = []
        student_boxes = []
        
        # FIRST PASS: Classify all persons (students) from COCO detections
        for det in coco_detections:
            label = det['label']
            if label.lower() in self.student_labels:
                bbox = det['bbox']
                conf = det['confidence']
                student_boxes.append(bbox)
                
                # Extract face region from detection bbox
                face_bbox = self._detect_face_bbox(frame, bbox)
                face_crop = None
                if face_bbox is not None:
                    fx1, fy1, fx2, fy2 = face_bbox
                    face_crop = frame[fy1:fy2, fx1:fx2]

                if face_crop is None or face_crop.size == 0:
                    face_crop = self._extract_face_region(frame, bbox, face_height_ratio=0.35, face_width_ratio=0.7)
                    face_bbox = self._get_face_bbox(frame, bbox, face_height_ratio=0.35, face_width_ratio=0.7)
                
                if face_crop is not None:
                    # Classify attention using EfficientNet or heuristic fallback
                    try:
                        # Check if using heuristic classifier
                        if hasattr(self.classifier, 'use_heuristic') and self.classifier.use_heuristic:
                            # Use heuristic classification
                            predicted_class, confidence = heuristic_classification(face_crop)
                        else:
                            # Use neural network
                            pil_image = Image.fromarray(face_crop[:, :, ::-1])
                            tensor_image = self.transform(pil_image).unsqueeze(0).to(DEVICE)
                            
                            # Run inference
                            with torch.no_grad():
                                output = self.classifier(tensor_image)
                                probabilities = torch.softmax(output, dim=1)
                                predicted_class = torch.argmax(probabilities, dim=1).item()
                                confidence = probabilities[0, predicted_class].item()
                        
                        status = self.labels[predicted_class]
                        
                        if confidence < 0.50:
                            status = 'unknown'

                        results.append({
                            'type': 'student',
                            'bbox': bbox,
                            'label': status,  # Show status (attentive/distracted) as the label
                            'status': status,
                            'confidence': confidence,
                            'detection_confidence': conf
                        })
                        
                        if return_crops:
                            crops.append(face_crop)
                    
                    except (RuntimeError, ValueError) as e:
                        print(f"⚠ Classification error for person at {bbox}: {e}")
                        results.append({
                            'type': 'student',
                            'bbox': bbox,
                            'label': 'unknown',  # Add label field for consistency
                            'face_bbox': face_bbox,
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
                        'label': 'unknown',  # Add label field for consistency
                        'face_bbox': None,
                        'status': 'unknown',
                        'confidence': 0.0,
                        'detection_confidence': conf,
                        'error': 'face_crop_failed'
                    })
        
        # SECOND PASS: Add all other COCO detections (phones, objects, etc.)
        # NO suppression - keep all detections
        for det in coco_detections:
            label = det['label']
            if label.lower() in self.student_labels:
                continue  # Already processed
            
            bbox = det['bbox']
            conf = det['confidence']
            
            # Categorize as phone or other object
            if label in self.phone_labels:
                results.append({
                    'type': 'phone',
                    'bbox': bbox,
                    'label': 'PHONE',  # Add label field
                    'status': 'phone_detected',
                    'object_label': label,
                    'confidence': conf,
                    'detection_confidence': conf
                })
            else:
                # Generic object
                results.append({
                    'type': 'object',
                    'bbox': bbox,
                    'label': label.upper(),  # Add label field
                    'status': label,
                    'object_label': label,
                    'confidence': conf,
                    'detection_confidence': conf
                })
        
        # THIRD PASS: Add all behavior detections (using mobile, etc.)
        # These come from the Track 1 model trained on this specific dataset
        for det in behavior_detections:
            label = det['label']
            bbox = det['bbox']
            conf = det['confidence']
            
            # Categorize behavior detections
            if label in {'using mobile', 'using_mobile'}:
                # Phone/mobile detected by behavior model
                results.append({
                    'type': 'phone',
                    'bbox': bbox,
                    'label': 'PHONE',  # Add label field
                    'status': 'using_mobile',
                    'object_label': label,
                    'confidence': conf,
                    'detection_confidence': conf,
                    'source': 'behavior_model'
                })
            else:
                # Other cheating behaviors
                results.append({
                    'type': 'behavior',
                    'bbox': bbox,
                    'label': label.upper(),  # Add label field
                    'status': label,
                    'object_label': label,
                    'confidence': conf,
                    'detection_confidence': conf,
                    'source': 'behavior_model'
                })
        
        if return_crops:
            return results, crops
        return results
