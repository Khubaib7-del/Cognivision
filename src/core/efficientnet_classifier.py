import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import os

class EfficientNetAttentionClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned on FER2013 for binary attention classification.
    Trained on Kaggle with 85.88% test accuracy.
    Maps FER2013 emotions to binary labels:
    - Attentive: happy (3), neutral (6)
    - Distracted: angry (0), disgust (1), fear (2), sad (4), surprise (5)
    """
    def __init__(self, num_classes=2):
        super(EfficientNetAttentionClassifier, self).__init__()
        
        # Load EfficientNet-B0 WITHOUT pre-trained weights (weights will be loaded from Kaggle checkpoint)
        self.model = models.efficientnet_b0(weights=None)
        
        # Replace classifier head for binary task
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def load_efficientnet_model(path=None, device='cpu'):
    """
    Load EfficientNet-B0 attention classifier.
    
    Args:
        path: Path to saved model weights (.pth file).
        device: Device to load model on ('cpu' or 'cuda').
    
    Returns:
        model: Loaded EfficientNetAttentionClassifier in eval mode.
    """
    model = EfficientNetAttentionClassifier(num_classes=2)
    
    if path and os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        
        # Handle checkpoint format mismatch: Kaggle saves without "model." prefix
        if any(k.startswith("model.") for k in state_dict.keys()):
            # Already has "model." prefix, load directly
            model.load_state_dict(state_dict)
        else:
            # No "model." prefix; need to add it
            new_state_dict = {"model." + k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        
        print(f"✓ EfficientNet weights loaded from: {path}")
    else:
        print(f"⚠ Warning: Weights file not found at {path}")
        print("  Using uninitialized model. Please provide trained weights.")
    
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model initialization
    print("Loading EfficientNet-B0 Attention Classifier...")
    model = load_efficientnet_model(device='cpu')
    print("✓ Model initialized successfully!")
    print(f"  Input size: 224x224 (RGB)")
    print(f"  Output classes: 2 (distracted, attentive)")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Forward pass output shape: {output.shape}")

