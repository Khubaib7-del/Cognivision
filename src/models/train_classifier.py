import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from attention_classifier import AttentionClassifier

def train_model():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 1 Training Device: {device}")

    # 2. Data Preparation (Phase 1 focus: Loading and Augmenting Data properly)
    data_dir = "data/raw"
    
    # We use 128x128 because we built a custom CNN from scratch without pre-trained weights.
    # Training large images from scratch requires massive computing, 128x128 is optimal for academic projects.
    img_size = 128
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5), # Data Augmentation (essential for train from scratch)
        transforms.RandomRotation(15),         
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        # Standard normalization for image channels
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        print(f"Dataset found. Classes: {dataset.classes}")
    except FileNotFoundError:
        print(f"ERROR: Dataset not found in {data_dir}. For Phase 1, you must place your training images here!")
        print("Expected structure:")
        print("  data/raw/attentive/... (many images)")
        print("  data/raw/distracted/... (many images)")
        return
        
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True) # Good batch size for custom CNN
    
    # 3. Initialize Custom Built Model
    print("Initializing Custom CNN Model from scratch...")
    model = AttentionClassifier(num_classes=len(dataset.classes))
    model.to(device)

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Using Adam optimizer (essential for training custom Deep Learning models from scratch)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    # 5. Training Pipeline (Phase 1 Demonstration)
    num_epochs = 30 
    print(f"Starting Phase 1 model training for {num_epochs} epochs with {len(dataset)} images...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate simple training accuracy for metrics logging (Good to show sir)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
        if avg_loss < 0.05: # Early exit condition if it converges perfectly
            print("Optimal academic threshold reached. Saving model phase 1.")
            break

    # 6. Save the Custom Trained Weights
    os.makedirs("src/models", exist_ok=True)
    save_path = "src/models/attention_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Phase 1 Success! Custom CNN weights saved to: {save_path}")

if __name__ == "__main__":
    train_model()
