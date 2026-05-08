# CogniVision - Deployment & Production Guide

## Project Status: READY FOR DEPLOYMENT ✓

### Summary
- **Phase 1**: ✓ Project refactored into modular architecture
- **Phase 2**: ✓ Model trained on Kaggle with 85.88% test accuracy
- **Phase 3**: ✓ Kaggle-trained EfficientNet-B0 integrated into pipeline
- **Phase 4**: ✓ End-to-end evaluation completed (100% success on local dataset)

---

## Quick Start

### 1. Run CLI (Webcam/Local Video)
```bash
python src/cli.py
```
- Opens OpenCV window with real-time detection and classification
- Shows bounding boxes with "attentive" or "distracted" labels
- Press 'q' to quit

### 2. Run API Server (Web Interface)
```bash
python src/api/app.py
```
- Starts FastAPI server on `http://localhost:8000`
- Access UI at `http://localhost:8000/`
- Video feed streams at `/video_feed`
- Server runs on port 8000 (configurable)

### 3. View Evaluation Report
```bash
# Latest report
cat logs/phase4_evaluation_*.json | jq .

# Or view in editor
code logs/phase4_evaluation_*.json
```

---

## Model Details

### EfficientNet-B0 Classifier
- **Architecture**: EfficientNet-B0 with binary classification head
- **Training Data**: FER2013 (28,709 images)
- **Test Accuracy**: 85.88%
- **Input Size**: 224×224 RGB
- **Output Classes**: 
  - Class 0: **Distracted** (angry, disgust, fear, sad, surprise)
  - Class 1: **Attentive** (happy, neutral)
- **Weights File**: `data/models/efficientnet_attention_best.pth`

### YOLO Detector
- **Model**: YOLOv8 Nano (`yolov8n.pt`)
- **Purpose**: Detects persons and cell phones in frame
- **Input**: Video frames (any size)
- **Output**: Bounding boxes with confidence scores

### Pipeline Flow
```
Video Frame → YOLOv8 Detection → Crop Face Region → 
EfficientNet Classification → Output (Attentive/Distracted)
```

---

## System Requirements

### Minimum
- Python 3.8+
- 2GB RAM
- CPU: Intel i5 or equivalent
- Optional: NVIDIA GPU for faster inference (CUDA 11.8+)

### Installation
```bash
pip install -r requirements.txt
```

### Supported OS
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 18.04+)

---

## Configuration

### File: `src/config.py`
Adjust these constants for your deployment:

```python
CLASSIFIER_INPUT_SIZE = 224      # Must match EfficientNet (224x224)
BATCH_SIZE = 32                  # For batch inference
LEARNING_RATE = 0.001            # (Not used in inference)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Environment Variables
```bash
# Optional: Set custom paths
export COGNIVISION_DATA_DIR=/path/to/data
export COGNIVISION_LOGS_DIR=/path/to/logs
```

---

## Performance Metrics

### Training Results (Kaggle)
| Metric | Value |
|--------|-------|
| Test Accuracy | 85.88% |
| Test Loss | 0.3308 |
| Best Val Accuracy | 87.20% (Epoch 12) |
| Best Val Loss | 0.5652 (Epoch 11) |

### Inference Speed (Local CPU)
- YOLOv8 Detection: ~50ms per frame
- EfficientNet Classification: ~30ms per detection
- Total latency: ~100ms per frame (10 FPS)

**GPU (NVIDIA T4)**: ~5x faster

---

## Troubleshooting

### Issue: Webcam not detected
```bash
# Check available cameras
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Issue: CUDA errors
```bash
# Fall back to CPU
export DEVICE=cpu
python src/cli.py
```

### Issue: Model weights not found
```bash
# Verify file exists
ls -la data/models/efficientnet_attention_best.pth

# If missing, download from Kaggle output
# Then copy to: data/models/efficientnet_attention_best.pth
```

### Issue: Out of memory
Reduce batch size or use GPU:
```python
# In src/config.py
BATCH_SIZE = 8  # Smaller batches
DEVICE = "cuda"  # Use GPU
```

---

## Deployment Options

### Option A: Standalone CLI
Best for: Local testing, demonstrations
```bash
python src/cli.py
```

### Option B: Web API (FastAPI)
Best for: Remote access, integration with other services
```bash
python src/api/app.py --host 0.0.0.0 --port 8000
```

### Option C: Docker Container
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/api/app.py"]
```

### Option D: Cloud Deployment (AWS/GCP/Azure)
1. Build Docker image
2. Push to container registry
3. Deploy to ECS/GKE/ACI

---

## API Endpoints

### FastAPI Server (`src/api/app.py`)

#### GET `/`
Returns HTML dashboard with live video feed.

#### GET `/video_feed`
Video stream endpoint (MJPEG format).
Use in `<img>` tag:
```html
<img src="http://localhost:8000/video_feed" />
```

#### POST `/detect` (Future)
Send image for detection:
```bash
curl -X POST -F "image=@test.jpg" http://localhost:8000/detect
```

---

## Output Format

### CLI Output
```
Frame: 123/500 | FPS: 9.5
Student 1: ATTENTIVE (0.92) @ (100,50,200,250)
Student 2: DISTRACTED (0.87) @ (350,100,450,300)
Phone Detected @ (400,120,420,140)
```

### API Response
```json
{
  "frame_id": 123,
  "timestamp": "2026-05-08T22:00:00Z",
  "detections": [
    {
      "type": "student",
      "status": "attentive",
      "confidence": 0.92,
      "bbox": [100, 50, 200, 250]
    },
    {
      "type": "distraction",
      "status": "phone_detected",
      "confidence": 0.95,
      "bbox": [400, 120, 420, 140]
    }
  ]
}
```

---

## Monitoring & Logging

### Log Files
```
logs/
├── phase4_evaluation_*.json    # Inference reports
├── app.log                     # API server logs
└── cli.log                     # CLI logs
```

### Enable Debug Logging
```python
# In src/logging_setup.py
logger.setLevel(logging.DEBUG)
```

---

## Next Steps

### For Development
1. Collect more training data (improve test accuracy)
2. Fine-tune hyperparameters
3. Test on different lighting conditions
4. Add analytics dashboard

### For Production
1. Deploy on cloud (AWS/GCP)
2. Set up monitoring and alerts
3. Configure CI/CD pipeline
4. Create admin dashboard

### For Research
1. Analyze misclassified samples
2. Improve feature extraction
3. Experiment with other architectures (MobileNet, ResNet)
4. Add temporal smoothing for video

---

## Support & Contact

For issues or questions:
1. Check `PROJECT_HANDOVER.md` for project context
2. Review `CONTRIBUTING.md` for development guidelines
3. See troubleshooting section above
4. Check logs in `logs/` directory

---

## License & Attribution

- **YOLOv8**: Ultralytics (Apache 2.0)
- **EfficientNet**: Weights from torchvision (Apache 2.0)
- **Training Data**: FER2013 (Academic Use)
- **Project**: CogniVision (Semester Project)

---

**Status**: Ready for Production ✓  
**Last Updated**: May 8, 2026  
**Model Test Accuracy**: 85.88%  
**Phase**: 4/4 Complete
