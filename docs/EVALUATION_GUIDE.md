# CogniVision: Evaluation & Teaching Guide 🎓

This document is designed to help you understand the "How" and "Why" of CogniVision for your project evaluation.

## 🏗️ Technical Architecture
### Why FastAPI?
- **Speed**: It's one of the fastest Python frameworks.
- **Async Support**: Crucial for serving real-time AI model results without blocking the server.

### Why YOLOv8?
- **State-of-the-Art**: It's the current industry standard for real-time object detection.
- **Multi-tasking**: It can detect faces and objects (like phones) simultaneously in one pass.

## 🧠 AI Components Explained
### 1. Transfer Learning (The "Shortcut")
Instead of training a model from scratch (which takes weeks and millions of images), we use **Pretrained Models** (ResNet/MobileNet) that already "know" how to see shapes and colors. We only "fine-tune" the last layer to recognize specific classroom behaviors.

### 2. Bounding Boxes & Confidence Scores
- **Bounding Box**: The rectangle around a student's face.
- **Confidence Score**: How sure the AI is about its prediction (e.g., 0.95 = 95% sure).

### 3. The Attention Metric
We calculate the **Attention Percentage** using:
$$Attention\% = \frac{\text{Attentive Students}}{\text{Total Students}} \times 100$$
This gives a clear, quantifiable metric for the teacher/evaluator to see the classroom's engagement at a glance.

## 🛠️ Key Viva Questions & Answers
**Q: How does the system handle multiple students?**
*A: We use YOLOv8 for "Global Detection" to find all faces in the frame, then we crop those faces and run our "Attention Classifier" on each individual crop.*

**Q: What is the benefit of using DAiSEE dataset?**
*A: DAiSEE is a specialized dataset for "Engagement Detection" in a video environment, making it much more accurate for classroom scenarios than a generic emotion dataset.*

**Q: Why a Web Dashboard instead of just a script?**
*A: A web dashboard makes the system "User-Friendly" and "Ready-to-Deploy," showing that we considered the end-user (the teacher) in our design.*
