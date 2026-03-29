import torch
import torch.nn as nn
from torchvision import models

class AttentionClassifier(nn.Module):
    """
    CogniVision Attention Classifier
    Uses Transfer Learning with a MobileNetV2 base to classify 
    student behavior as 'Attentive' or 'Distracted'.
    """
    def __init__(self, num_classes=2):
        super(AttentionClassifier, self).__init__()
        # Load pretrained MobileNetV2 with modern weight handling
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        self.base_model = models.mobilenet_v2(weights=weights)
        
        # Freeze base layers for stability on small datasets
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Replace the classifier head
        # MobileNetV2 last layer is self.base_model.classifier[1]
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.base_model(x)

def load_model(path=None):
    model = AttentionClassifier()
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == "__main__":
    # Test model initialization
    model = AttentionClassifier()
    print(f"Model initialized with classification head:\n{model.base_model.classifier}")
