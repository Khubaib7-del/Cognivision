# CogniVision: Project Handover & Context

**Purpose of this file:** Provide absolute context to AI coding assistants in future sessions. 
*AI INSTRUCTION: If you are reading this, absorb the architecture, hardware limitations, and current state before making any code suggestions.*

---

## 1. Project Architecture
*   **Application Type:** FastAPI Web Dashboard (`127.0.0.1:8001`) rendering a real-time OpenCV webcam feed via exact HTML/CSS UI (`src/api/index.html`).
*   **AI Pipeline:** (`src/core/engine.py`)
    1.  **Detector:** YOLOv8 Nano (`yolov8n.pt`). Detects persons (students) and objects (phones/distractions).
    2.  **Classifier:** Custom PyTorch MobileNetV2 (`attention_model.pth`). Takes cropped face arrays and classifies them as `attentive` or `distracted`.
*   **Scoring Logic:** (`src/core/scorer.py`) Calculates an overall percentage of class attention and applies penalties for detected objects (like phones).

## 2. Hardware Environment & Limitations
*   **Hardware:** Windows OS, Laptop **CPU Only** (No dedicated GPU available).
*   **Interpreter:** Python 3.14.2 (Inside `.venv`).
*   **Known Bugs/Quirks Solved:**
    *   *Webcam Hanging:* Using experimental OpenCV flags like `cv2.CAP_DSHOW` caused terminal hangs for this specific hardware. We **reverted** to a standard `cv2.VideoCapture(0)`.
    *   *AI Loading Lag:* Booting PyTorch and YOLOv8 on a CPU takes 30-60 seconds before the webcam feed opens. *Do not assume the code is broken if it stalls on boot.*

## 3. Current Code State
*   **State:** Clean. Reverted precisely to the last active GitHub commit. 
*   **UI Status:** Individual bounding boxes display `LABEL: STATUS (XX.X%)`. A global `CLASS ATTENTION: 100%` is overlaid at the top left.
*   *Note:* The bounding boxes currently draw separate "Object" boxes over students holding phones (resulting in 2 overlapping boxes per person).

## 4. Next Session Roadmap
1.  **Retraining the Model:** The current MobileNetV2 classifier confidently hovers around `~73%` (Outputting Softmax logic). The primary goal is to **retrain this model using standard datasets** to achieve 90%+ confidence.
2.  **Addressing CPU Lag / Low FPS:** After replacing the model, implement **Frame Skipping** (e.g., executing AI on every 3rd frame but drawing the UI on every frame) or **Frame Resizing** (`640x480`) to drastically increase real-time webcam smoothness on the CPU.
3.  **UI Redesign:** Fix the "double box" issue (by merging proximity bounding boxes) only *after* the new models are fully integrated.
