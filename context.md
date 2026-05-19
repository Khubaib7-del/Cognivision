# CogniVision: Comprehensive Project Context for LLMs

## 1. Project Overview
**CogniVision** is an AI-powered system designed for automated student monitoring in educational environments (exams/classrooms). It aims to detect student attention levels and cheating-related behaviors in real-time.

### Core Problem
Manual invigilation is unscalable and prone to human error. CogniVision provides an objective monitoring layer to flag suspicious patterns.

---

## 2. Technical Stack
- **Languages**: Python 3.x
- **Deep Learning Framework**: PyTorch (for classification)
- **Object Detection**: Ultralytics YOLOv8 (v8n variant)
- **Backend/API**: FastAPI
- **Image Processing**: OpenCV, PIL
- **Data Handling**: Pandas, NumPy

---

## 3. System Architecture (The Pipeline)
The system uses a **multi-track dual-detector pipeline**:

1.  **Person Detection**: A COCO-pretrained YOLOv8n model identifies 'person' objects in the frame.
2.  **Face Localization**: For each detected person, the system extracts a face crop using a combination of a Haar Cascade classifier (for precision) and a geometric heuristic (top 35% of the body box).
3.  **Attention Classification (Track 1)**:
    -   **Model**: EfficientNet-B2 (fine-tuned on FER2013 data).
    -   **Task**: Binary classification (Attentive vs. Distracted).
    -   **Input**: 224x224 RGB face crop.
4.  **Behavior Detection (Track 2)**:
    -   **Model**: Custom-trained YOLOv8n.
    -   **Classes**: `leaning_to_copy`, `looking_around`, `sharing_answers`, `using_mobile`.
    -   **Inference**: Runs in parallel with Track 1.
5.  **Output Merging**: Detections from all tracks are merged. **Crucially, no cross-detector suppression is applied** to ensure behavior detections (like a phone) aren't erased by person detections.

---

## 4. Datasets & Training

### RGB Attention Dataset
-   **Source**: FER2013 (Facial Emotion Recognition 2013) dataset, mapped to binary labels.
-   **Classes**: Attentive (0), Distracted (1).
-   **Augmentation**: Horizontal flip, rotation (10°), color jitter.
-   **Normalization**: ImageNet standards (mean/std).

### Behavior Dataset
-   **Size**: 656 custom-labeled images.
-   **Split**: 480 Train / 176 Validation.
-   **Format**: YOLO normalized bounding boxes.

---

## 5. Model Implementations

### EfficientNet-B2 (Classifier)
-   **Architecture**: Compound-scaled CNN using **MBConv blocks**.
-   **Hidden Layers**: Includes **Squeeze-and-Excitation (SE)** blocks for channel-wise attention.
-   **Custom Head**:
    - `GlobalAveragePooling2d`
    - `Dropout(0.3)`
    - `Linear(1408, 2)`
-   **Parameters**: ~9.2M.
-   **Optimizer**: AdamW with `ReduceLROnPlateau` scheduler.
-   **Loss**: CrossEntropyLoss.

### YOLOv8n (Detector)
-   **Architecture**: CSPDarknet backbone with a decoupled head.
-   **Neck**: PAN-FPN for multi-scale feature fusion.
-   **Parameters**: ~3.2M.
-   **Training Specs**: 100 epochs, 960px image size, batch size 16.

### Custom Scratch CNN
-   **Architecture**: 4x (Conv2d -> BatchNorm -> ReLU -> MaxPool).
-   **Hidden Layers**: Dense layer with 512 neurons + Dropout (0.5).
-   **Parameters**: ~4.5M.
-   **Purpose**: Demonstrates baseline understanding of CNN architecture without pre-trained weights.

---

## 6. Hyperparameters & Configuration
- **Learning Rates**: 1e-3 (Attention), 3e-4 (YOLO).
- **Batch Size**: 32.
- **Input Resolutions**: 224x224 (EfficientNet), 128x128 (Scratch), 960x960 (YOLO).
- **Dropout Rate**: 0.3 to 0.5 (varies by model).

---

## 7. Performance Metrics

### RGB Attention Classifier
-   **Validation Accuracy**: 90.10%
-   **Test Accuracy**: 89.05%
-   **Precision**: 89.2%
-   **Recall**: 88.8%
-   **F1-Score**: 0.890

### YOLO Behavior Detector
-   **mAP@0.5**: 37.3%
-   **Confidence Threshold**: 0.25 (tuned to prioritize recall/safety).

---

## 7. Key Engineering Challenges & Fixes

### The "Suppression Bug"
-   **Issue**: Early versions used strict IoU-based suppression across detectors. This caused valid objects (like a phone held by a student) to be suppressed by the student's own bounding box.
-   **Fix**: Restructured the pipeline to allow overlapping detections from different detector tracks.

### Fallback Mechanism
-   **Heuristic Classifier**: In environments where model weights (`.pth`) fail to load, the system falls back to a grayscale-based heuristic that analyzes eye-region brightness and face symmetry to estimate attention.

---

## 8. Ethical & Practical Constraints
-   **Intent vs. Action**: The model detects visual patterns, not malicious intent.
-   **Bias**: Performance may vary based on lighting, camera angle, and demographics (inherent to FER2013 and custom small-batch datasets).
-   **Privacy**: System is designed for real-time monitoring, requiring careful handling of student data.
