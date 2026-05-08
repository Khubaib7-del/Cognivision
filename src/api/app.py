from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import threading
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipeline import ProcessingPipeline
from src.core.scorer import CogniVisionScorer
from src.logging_setup import setup_logging

logger = setup_logging("api")

app = FastAPI(
    title="CogniVision Dashboard",
    description="Real-time classroom attention monitoring",
    version="1.0.0"
)

# Initialize pipeline and scorer
try:
    pipeline = ProcessingPipeline()
    scorer = CogniVisionScorer()
    logger.info("✓ Pipeline and Scorer initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize: {e}")
    pipeline = None
    scorer = None

class VideoStreamer:
    """Manages video capture and streaming."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        logger.info("✓ Video capture initialized")
        
    def get_frame(self):
        """Generate MJPEG frames."""
        while True:
            with self.lock:
                success, frame = self.cap.read()
            
            if not success:
                logger.warning("Failed to read frame from webcam")
                break
            
            try:
                # Run pipeline
                detections = pipeline.process_frame(frame)
                class_score = scorer.calculate_class_score(detections)
                
                # Draw detections
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    status = det['status']
                    conf = det['confidence']
                    det_type = det['type']
                    
                    if det_type == 'student':
                        color = (0, 255, 0) if status == 'attentive' else (0, 0, 255)
                    else:
                        color = (0, 165, 255)
                    
                    label_text = f"{det_type.upper()}: {status} ({conf:.1%})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Overlay score
                cv2.putText(frame, f"CLASS ATTENTION: {class_score}%", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                
                # Encode as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue

streamer = VideoStreamer()

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main dashboard HTML."""
    html_file = Path(__file__).parent / "index.html"
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Dashboard HTML not found</h1>"

@app.get("/video_feed")
async def video_feed():
    """Stream MJPEG video feed."""
    return StreamingResponse(
        streamer.get_frame(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stats")
async def get_stats():
    """Return system status."""
    return {
        "status": "online",
        "version": "1.0.0",
        "device": "cpu"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
