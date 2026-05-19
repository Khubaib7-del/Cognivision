import torch
import torch.nn as nn
from torchvision import models
import os

class EfficientNetAttentionClassifier(nn.Module):
    """
    EfficientNet-based attention classifier.
    Falls back to heuristic classification if model weights not available.
    """
    def __init__(self, num_classes=2, variant='b2', use_pretrained=False):
        super(EfficientNetAttentionClassifier, self).__init__()
        
        # Build lightweight model - no pretrained weights to avoid downloads
        if variant == 'b2':
            self.model = models.efficientnet_b2(weights=None)
        elif variant == 'b0':
            self.model = models.efficientnet_b0(weights=None)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
        
        # Replace classifier head for binary task
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def heuristic_classification(face_crop):
    """
    Fallback heuristic classifier using face features.
    
    Analyzes:
    - Face brightness patterns (dark eyes indicate distraction)
    - Horizontal edge distribution (head tilt indicators)
    - Vertical symmetry (attention-related)
    
    Returns: (class_id, confidence)
        0 = ATTENTIVE (looking forward, eyes open, symmetric)
        1 = DISTRACTED (looking down/away, eyes closed, asymmetric)
    """
    if face_crop is None or face_crop.size == 0:
        return 0, 0.5
    
    # Convert to grayscale
    if len(face_crop.shape) == 3:
        gray = (0.114 * face_crop[:, :, 0] + 0.587 * face_crop[:, :, 1] + 0.299 * face_crop[:, :, 2]).astype('uint8')
    else:
        gray = face_crop
    
    h, w = gray.shape
    
    # Feature 1: Brightness distribution in upper half (eyes region)
    upper_half = gray[:h//2, :]
    eye_brightness = upper_half.mean()
    
    # Feature 2: Edge detection in face
    # Compute horizontal edges (indicates head position/tilt)
    edges_h = 0
    if h > 1:
        edges_h = abs(gray[1:, :].astype(int) - gray[:-1, :].astype(int)).mean()
    
    # Feature 3: Left-right symmetry (attentive = more symmetric)
    left_half = gray[:, :w//2].mean()
    right_half = gray[:, w//2:].mean()
    symmetry = 1.0 - (abs(left_half - right_half) / (left_half + right_half + 1e-6))
    
    # Feature 4: Face center intensity (determines if looking at camera)
    center_region = gray[h//4:3*h//4, w//4:3*w//4]
    center_intensity = center_region.mean()
    
    # Heuristic scoring:
    # - Bright eyes + high center intensity + symmetric = ATTENTIVE (0)
    # - Dark regions + low center intensity + asymmetric = DISTRACTED (1)
    
    score = 0
    score += (eye_brightness / 255.0) * 0.3  # bright eyes favor attentive
    score += (center_intensity / 255.0) * 0.3  # bright center favor attentive
    score += symmetry * 0.2  # symmetry favors attentive
    score += (1.0 - edges_h / 50.0) * 0.2  # low edges favor attentive
    
    score = max(0, min(1, score))  # Clamp to [0, 1]
    
    # Decision threshold: 0.5 middle
    # > 0.55 = ATTENTIVE (0), < 0.45 = DISTRACTED (1), else uncertain
    if score > 0.55:
        return 0, score  # attentive
    elif score < 0.45:
        return 1, 1.0 - score  # distracted
    else:
        # Uncertain - use random bias toward attentive (default positive)
        return 0, 0.5 + (score - 0.45) * 2  # smooth transition


def load_efficientnet_model(path=None, device='cpu'):
    """
    Load EfficientNet attention classifier with fallback to heuristic.
    
    Args:
        path: Path to saved model weights (.pth file).
        device: Device to load model on ('cpu' or 'cuda').
    
    Returns:
        model: Loaded EfficientNetAttentionClassifier in eval mode.
    """
    model = None
    loaded_from_checkpoint = False
    
    if path and os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=device)
            
            # Handle checkpoint format mismatch
            candidate_state_dict = state_dict
            if not any(k.startswith("model.") for k in state_dict.keys()):
                candidate_state_dict = {"model." + k: v for k, v in state_dict.items()}
            
            # Try loading with preferred variants
            for variant in ['b2', 'b0']:
                candidate_model = EfficientNetAttentionClassifier(num_classes=2, variant=variant, use_pretrained=False)
                try:
                    candidate_model.load_state_dict(candidate_state_dict)
                    model = candidate_model
                    loaded_from_checkpoint = True
                    print(f"✓ EfficientNet-{variant.upper()} weights loaded from: {path}")
                    break
                except RuntimeError:
                    continue
            
            if not loaded_from_checkpoint:
                print(f"⚠ Could not load checkpoint from {path}, using heuristic classifier")
        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
    
    if model is None:
        print(f"✓ Using heuristic attention classifier (no checkpoint weights)")
        if path:
            print(f"  (Checkpoint file: {path})")
        model = EfficientNetAttentionClassifier(num_classes=2, variant='b2', use_pretrained=False)
        model.use_heuristic = True  # Mark to use heuristic inference
    else:
        model.use_heuristic = False
    
    model = model.to(device)
    model.eval()
    return model



if __name__ == "__main__":
    # Test model initialization
    print("Loading EfficientNet Attention Classifier...")
    model = load_efficientnet_model(device='cpu')
    print("✓ Model initialized successfully!")
    print(f"  Input size: 224x224 (RGB)")
    print(f"  Output classes: 2 (distracted, attentive)")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Forward pass output shape: {output.shape}")

