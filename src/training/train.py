import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path

from src.core.classifier import AttentionClassifier
from src.config import (
    RAW_DATA_DIR, CHECKPOINTS_DIR, BATCH_SIZE, 
    LEARNING_RATE, NUM_EPOCHS, CLASSIFIER_INPUT_SIZE,
    DEVICE
)

def train_model():
    """
    Train the attention classifier model with validation tracking.
    """
    print(f"Training Device: {DEVICE}")
    
    # Data Preparation
    img_size = CLASSIFIER_INPUT_SIZE
    
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),         
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(str(RAW_DATA_DIR), transform=train_transform)
        print(f"✓ Dataset loaded. Classes: {dataset.classes}")
        print(f"✓ Total images: {len(dataset)}")
    except FileNotFoundError:
        print(f"✗ ERROR: Dataset not found in {RAW_DATA_DIR}")
        print("Expected structure:")
        print(f"  {RAW_DATA_DIR}/attentive/...")
        print(f"  {RAW_DATA_DIR}/distracted/...")
        return
        
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    print("Initializing Custom CNN Model...")
    model = AttentionClassifier(num_classes=len(dataset.classes))
    model.to(DEVICE)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    # Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = CHECKPOINTS_DIR / f"attention_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")
        
        if avg_loss < 0.05:
            print("✓ Optimal threshold reached. Stopping early.")
            break

    # Save final model
    final_path = CHECKPOINTS_DIR.parent / "attention_classifier_best.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Training complete! Model saved to: {final_path}")

if __name__ == "__main__":
    train_model()
