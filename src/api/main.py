from fastapi import FastAPI, Response, Query
from fastapi.responses import StreamingResponse
import cv2
import sys
import os
import threading
import time
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import File, UploadFile
from src.core.pipeline import ProcessingPipeline
from src.core.scorer import CogniVisionScorer
from src.config import CLASSIFIER_MODEL

app = FastAPI(title="CogniVision Dashboard")

# Initialize AI Pipeline and Scorer with trained classifier
pipeline = ProcessingPipeline(classifier_weights=CLASSIFIER_MODEL)
scorer = CogniVisionScorer()

class VideoStreamer:
    def __init__(self):
        self.lock = threading.Lock()

    def _placeholder_frame(self, message: str = "Use Local Webcam"):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (639, 479), (45, 45, 90), thickness=-1)
        cv2.putText(frame, "CogniVision", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (129, 140, 248), 3)
        cv2.putText(frame, message, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Click 'Use Local Webcam' button to start", (40, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 220), 2)
        return frame
        
    def get_frame(self):
        while True:
            frame = self._placeholder_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1.0)

streamer = VideoStreamer()

@app.get("/")
async def index():
    from fastapi.responses import HTMLResponse
    with open("src/api/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(streamer.get_frame(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
async def get_stats():
    return {"status": "online"}

@app.post("/api/report")
async def report():
    """Generate report endpoint."""
    return {
        "status": "success",
        "message": "Report generated (database integration coming soon)",
        "timestamp": time.time()
    }

@app.post("/api/reset")
async def reset():
    """Reset engine endpoint."""
    return {
        "status": "success",
        "message": "Pipeline reset",
        "timestamp": time.time()
    }
@app.post('/api/infer')
async def infer_image(file: UploadFile = File(...), confidence_threshold: float = Query(0.5, ge=0.0, le=1.0)):
    """Accepts an uploaded image and returns detection + score JSON.
    Useful for web UI or remote clients that send frames for inference.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid_image"}

    detections = pipeline.process_frame(img)
    filtered_detections = [
        detection
        for detection in detections
        if detection.get("confidence", 0.0) >= confidence_threshold
    ]
    class_score = scorer.calculate_class_score(filtered_detections)

    return {
        "detections": filtered_detections,
        "class_score": class_score,
        "confidence_threshold": confidence_threshold,
    }


if __name__ == "__main__":
    import uvicorn
    # Changed to 8001 to avoid port conflicts on Windows
    uvicorn.run(app, host="0.0.0.0", port=8001)
