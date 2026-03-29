from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import sys
import os
import threading
import time

# Add src to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import CogniVisionEngine
from core.scorer import CogniVisionScorer

app = FastAPI(title="CogniVision Dashboard")

# Initialize AI Engine and Scorer
engine = CogniVisionEngine()
scorer = CogniVisionScorer()

class VideoStreamer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        
    def get_frame(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            # Run AI Engine
            detections = engine.process_frame(frame)
            class_score = scorer.calculate_class_score(detections)
            
            # Draw on frame
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                status = det['status']
                conf = det['confidence']
                color = (0, 255, 0) if status == 'attentive' else (0, 0, 255)
                
                label_text = f"{det['type'].upper()}: {status} ({conf:.1%})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Overlay Score
            cv2.putText(frame, f"ATTENTION: {class_score}%", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
    # Placeholder for historical stats if needed later
    return {"status": "online"}

if __name__ == "__main__":
    import uvicorn
    # Changed to 8001 to avoid port conflicts on Windows
    uvicorn.run(app, host="0.0.0.0", port=8001)
