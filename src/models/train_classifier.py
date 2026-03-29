import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

from attention_classifier import AttentionClassifier

def train_model():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Data Preparation (With Advanced Augmentation)
    data_dir = "data/raw"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # Effectively doubles your data
        transforms.RandomRotation(15),         # Handles head tilts
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handles lighting issues
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Initialize Model
    model = AttentionClassifier(num_classes=len(dataset.classes))
    model.to(device)

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002) # Adjusted LR
    
    # 5. Training Loop
    num_epochs = 30 # Even more epochs for better convergence
    print(f"Starting training for {num_epochs} epochs with the new {len(dataset)} images...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            if avg_loss < 0.1: # Early stop if we overfit well
                print("Optimal loss reached. Saving model.")
                break

    # 6. Save Model
    save_path = "src/models/attention_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Success! Model weights saved to: {save_path}")

if __name__ == "__main__":
    train_model()
