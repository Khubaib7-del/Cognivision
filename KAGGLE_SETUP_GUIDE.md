# CogniVision Training on Kaggle - Step by Step Guide

## WORKFLOW OVERVIEW

You have 3 options:

### **FASTEST: Option C (Copy-Paste, No GitHub needed)**
✅ **Recommended** - Creates self-contained notebook, works immediately

Important: once you click Run All in Kaggle, the notebook runs on Kaggle's cloud compute. You can close your laptop after the run starts, as long as you do not manually stop the session.

Steps:
1. Go to kaggle.com/notebooks/create
2. Click "Add Data" → Search "fer2013" → Add "Facial Expression Recognition (FER) 2013" dataset
3. In notebook, copy code from section below
4. Run cells one by one
5. Download trained model files to your local project

---

### **OPTION A: Via GitHub (For future)**
Steps:
1. Push your CogniVision repo to GitHub (git push origin main)
2. In Kaggle Notebook cell:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/cognivision.git
   %cd cognivision
   ```
3. Install requirements:
   ```bash
   !pip install -r requirements.txt
   ```
4. Run training:
   ```bash
   !python src/training/train_efficientnet_kaggle.py --epochs 12 --batch-size 64
   ```

---

### **OPTION B: Kaggle Dataset Upload**
Steps:
1. Create Kaggle Dataset with your repo files zipped
2. Add dataset to notebook
3. Unzip and reference files

---

## COPY THIS CODE INTO YOUR KAGGLE NOTEBOOK (OPTION C)

### Cell 1: Check GPU & Install Packages
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# Install required packages
!pip install -q efficientnet-pytorch
```

### Cell 2: Load FER2013 Dataset
```python
import pandas as pd
import os
from pathlib import Path

competition_files = [
    "/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv",
    "/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv",
    "/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013.csv",
]

fer2013_path = None
for candidate in competition_files:
    candidate_path = Path(candidate)
    if candidate_path.exists():
        fer2013_path = str(candidate_path)
        break

# Fallback: search all mounted Kaggle inputs for a FER2013 CSV-like file
if fer2013_path is None:
    for pattern in ("train.csv", "icml_face_data.csv", "fer2013.csv"):
        for path in Path("/kaggle/input").rglob(pattern):
            fer2013_path = str(path)
            break
        if fer2013_path is not None:
            break

# Final fallback: any CSV file inside the competition folder
if fer2013_path is None:
    for path in Path("/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge").rglob("*.csv"):
        fer2013_path = str(path)
        break

if fer2013_path is None:
    raise FileNotFoundError(
        "FER2013 CSV not found under /kaggle/input. "
        "Add the 'Challenges in Representation Learning: Facial Expression Recognition Challenge' competition input."
    )

print(f"Using FER2013 file: {fer2013_path}")

df = pd.read_csv(fer2013_path)
print(f"FER2013 Dataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Emotions: {df['emotion'].unique()}")
print(f"\nSample:")
print(df.head())
```

### Cell 3: Define Binary Conversion & Dataset Class
```python
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

# FER2013 emotion IDs
# 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
ATTENTIVE_IDS = {3, 6}  # happy, neutral
DISTRACTED_IDS = {0, 1, 2, 4, 5}  # angry, disgust, fear, sad, surprise

class Fer2013Dataset(Dataset):
    """Convert FER2013 to binary: attentive=1, distracted=0"""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def pixels_to_image(pixel_string):
        values = np.fromstring(pixel_string, dtype=np.uint8, sep=" ")
        img = values.reshape(48, 48)
        # Avoid deprecated Pillow 'mode' parameter; infer mode from array then convert to RGB
        pil_img = Image.fromarray(img).convert("RGB")
        return pil_img
    
    @staticmethod
    def emotion_to_binary(emotion_id):
        if emotion_id in ATTENTIVE_IDS:
            return 1  # attentive
        elif emotion_id in DISTRACTED_IDS:
            return 0  # distracted
        else:
            return -1  # skip
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.pixels_to_image(row["pixels"])
        label = self.emotion_to_binary(int(row["emotion"]))
        
        if label == -1:  # Skip unknown emotions
            # Return dummy instead
            label = 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def stratified_split(dataframe, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """Split by emotion label so each split keeps a similar class mix."""
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in dataframe.groupby("emotion", sort=False):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_total = len(group)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df

# Split data
if "Usage" in df.columns:
    train_df = df[df["Usage"] == "Training"].copy()
    val_df = df[df["Usage"] == "PublicTest"].copy()
    test_df = df[df["Usage"] == "PrivateTest"].copy()
else:
    train_df, val_df, test_df = stratified_split(df, train_frac=0.8, val_frac=0.1, test_frac=0.1)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

### Cell 4: Create DataLoaders
```python
# Transform pipelines
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Create datasets
train_dataset = Fer2013Dataset(train_df, transform=train_transform)
val_dataset = Fer2013Dataset(val_df, transform=eval_transform)
test_dataset = Fer2013Dataset(test_df, transform=eval_transform)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

### Cell 5: Build Model (EfficientNet)
```python
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained EfficientNet-B0
weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Replace classifier for binary task
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)  # 2 classes: attentive, distracted

model = model.to(device)
print(f"Model moved to {device}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
```

### Cell 6: Setup Training
```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

print("✓ Training setup complete")
```

### Cell 7: Training Loop
```python
from pathlib import Path
import json

OUTPUT_DIR = Path("/kaggle/working/cognivision_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = OUTPUT_DIR / "efficientnet_attention_best.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "efficientnet_attention_last.pth"
STATE_PATH = OUTPUT_DIR / "training_state.json"

def run_epoch(model, loader, criterion, optimizer, device, train=False):
    if train:
        model.train()
    else:
        model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if train:
                optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if train:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

# Train for 12 epochs
epochs = 12
best_val_loss = float('inf')
history = []

try:
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        
        scheduler.step(val_loss)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")
        
        epoch_checkpoint = OUTPUT_DIR / f"efficientnet_attention_epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_checkpoint)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ Best model saved to {BEST_MODEL_PATH}")

        with open(STATE_PATH, "w") as f:
            json.dump({
                "epoch": epoch,
                "epochs": epochs,
                "best_val_loss": float(best_val_loss),
                "history": history,
                "last_checkpoint": str(epoch_checkpoint),
                "best_model": str(BEST_MODEL_PATH),
            }, f, indent=2)

        print(f"  ✓ Checkpoint saved to {epoch_checkpoint}")
        print(f"  ✓ Training state saved to {STATE_PATH}")

except KeyboardInterrupt:
    # Save best-effort last model & state so you can resume later
    curr_epoch = locals().get('epoch', 0)
    print("\nTraining interrupted by user — saving last checkpoint and state...")
    torch.save(model.state_dict(), LAST_MODEL_PATH)
    with open(STATE_PATH, "w") as f:
        json.dump({
            "epoch": curr_epoch,
            "epochs": epochs,
            "best_val_loss": float(best_val_loss),
            "history": history,
            "last_checkpoint": str(LAST_MODEL_PATH),
            "best_model": str(BEST_MODEL_PATH),
        }, f, indent=2)
    print(f"  ✓ Saved last model to {LAST_MODEL_PATH}")
    print(f"  ✓ Training state saved to {STATE_PATH}")
    raise
else:
    torch.save(model.state_dict(), LAST_MODEL_PATH)
    print(f"\n✓ Training complete! Last model saved to {LAST_MODEL_PATH}")
```

### Cell 8: Test & Save Metrics
```python
import json
from pathlib import Path

# Final test evaluation
# If a best checkpoint was saved, load it before final evaluation
if BEST_MODEL_PATH.exists():
    print(f"Loading best model from {BEST_MODEL_PATH} for final evaluation")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)

test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Save metrics
metrics_path = Path("/kaggle/working/cognivision_outputs/metrics.json")
metrics = {
    "test_loss": float(test_loss),
    "test_acc": float(test_acc),
    "history": history,
    "label_map": {"0": "distracted", "1": "attentive"}
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Files saved:")
print(f"  - {BEST_MODEL_PATH}")
print(f"  - {LAST_MODEL_PATH}") 
print(f"  - {metrics_path}")
```

---

**Quick reminder — Commit & Save on Kaggle**

- After verifying cells run correctly, use Kaggle's `Commit & Run` to start the job.
- Once the run starts, use the notebook `Save` (snapshot) option so the session continues in the cloud.
- Monitor progress under the **Work** sidebar → **Outputs** and check `metrics.json` and saved `.pth` files when finished.

---

## AFTER TRAINING ON KAGGLE

Download the 3 files from Kaggle output:
1. efficientnet_attention_best.pth
2. efficientnet_attention_last.pth
3. metrics.json

Move them to your local project:
```
efficientnet_attention_best.pth → data/models/
metrics.json → logs/
```

Then tell me: **"I have trained the model"**

And provide:
- Test accuracy %
- Any issues encountered

---

## GITHUB WORKFLOW (For Reference)

If you want to use GitHub:

```bash
# Local: Add & commit changes
git add -A
git commit -m "feat: add kaggle training pipeline"
git push origin main

# In Kaggle Notebook:
!git clone https://github.com/YOUR_USERNAME/cognivision.git
%cd cognivision
!pip install -r requirements.txt
!python src/training/train_efficientnet_kaggle.py --epochs 12
```

Then download outputs back to local project.

---

## WHICH OPTION TO USE?

| Option | Setup Time | Future Updates | Best For |
|--------|-----------|-----------------|----------|
| A (GitHub) | 10 min | ✓ Easy (pull updates) | Long-term projects |
| B (Dataset) | 15 min | ✗ Slow | One-time runs |
| **C (Copy-Paste)** | **5 min** | ✓ Manual | **You NOW** |

**→ Use Option C NOW, upgrade to A later if needed**
