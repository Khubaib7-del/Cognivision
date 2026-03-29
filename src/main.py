import cv2
import sys
import os

# Add src to python path for internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import CogniVisionEngine
from core.scorer import CogniVisionScorer

def main():
    print("--- CogniVision: Live Multi-Student Monitoring ---")
    
    # 1. Initialize Components
    try:
        engine = CogniVisionEngine()
        scorer = CogniVisionScorer()
        print("AI Engine & Scorer: READY")
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize AI pipeline: {e}")
        return

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Process with AI Engine
        detections = engine.process_frame(frame)
        class_score = scorer.calculate_class_score(detections)

        # 4. Visualization
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            status = det['status']
            color = (0, 255, 0) if status == 'attentive' else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det['type'].upper()}: {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay overall score
        cv2.putText(frame, f"CLASS ATTENTION: {class_score}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        cv2.imshow("CogniVision Dashboard (Phase 2)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
