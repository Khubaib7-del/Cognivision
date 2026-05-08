import torch
import torch.nn as nn
import os

class AttentionClassifier(nn.Module):
    """
    CogniVision Custom Attention Classifier
    A Convolutional Neural Network (CNN) built entirely from scratch 
    to classify student behavior as 'Attentive' or 'Distracted'.
    This architecture uses NO pre-trained weights, avoiding models like MobileNet.
    """
    def __init__(self, num_classes=2):
        super(AttentionClassifier, self).__init__()
        
        # Phase 1: Custom Feature Extraction (Convolutional Layers)
        # Input: 3 Channels (RGB), resized to 128x128 pixels in the training pipeline
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 x 64 x 64 (from 128x128 input)
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 32 x 32
            
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x 16 x 16
            
            # Block 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128 x 8 x 8 = 8192 parameters
        )
        
        # Phase 1: Classification Head (Dense / Fully Connected Layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), 
            nn.ReLU(),
            nn.Dropout(0.5), # Standard dropout to prevent overfitting on custom dataset
            nn.Linear(512, num_classes)
            # Softmax is excluded here because PyTorch's CrossEntropyLoss handles it natively during training.
            # During inference API, we can apply softmax if we need raw probabilities.
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(path=None):
    model = AttentionClassifier()
    if path and os.path.exists(path):
        # We use map_location='cpu' so the model loads robustly on local laptops without a strong GPU
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

if __name__ == "__main__":
    # Test model initialization and forward pass
    model = AttentionClassifier()
    print("Custom PyTorch CNN Initialized successfully!")
    print(model)
    
    # Test with a dummy tensor (Batch Size 1, Channels 3, 128x128 pixels)
    print("\nSimulating a forward pass with a dummy 128x128 image...")
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"Output shape (Batch Size, Classes): {output.shape}")
