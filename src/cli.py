import cv2
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.pipeline import ProcessingPipeline
from src.core.scorer import CogniVisionScorer
from src.logging_setup import setup_logging

logger = setup_logging("cli")

def main():
    """
    CLI mode: Real-time webcam monitoring with OpenCV display.
    """
    logger.info("=== CogniVision: Live Multi-Student Monitoring (CLI) ===")
    
    # 1. Initialize Components
    try:
        pipeline = ProcessingPipeline()
        scorer = CogniVisionScorer()
        logger.info("✓ Pipeline & Scorer initialized")
    except Exception as e:
        logger.error(f"✗ FATAL: Failed to initialize pipeline: {e}")
        return

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("✗ Could not access webcam")
        return
    
    logger.info("✓ Webcam opened")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 3. Process with Pipeline
            detections, crops = pipeline.process_frame(frame, return_crops=True)
            class_score = scorer.calculate_class_score(detections)

            # 4. Visualization
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                status = det['status']
                conf = det['confidence']
                det_type = det['type']
                
                if det_type == 'student':
                    color = (0, 255, 0) if status == 'attentive' else (0, 0, 255)
                else:  # distraction
                    color = (0, 165, 255)  # Orange for phone
                
                label_text = f"{det_type.upper()}: {status} ({conf:.1%})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show first face crop in debug window
            if crops:
                debug_face = cv2.resize(crops[0], (200, 200))
                cv2.imshow("CogniVision: Face Debug View", debug_face)

            # Overlay overall score
            cv2.putText(frame, f"CLASS ATTENTION: {class_score}%", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            cv2.imshow("CogniVision Dashboard (CLI)", frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("✓ Monitoring stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✓ Resources released")

if __name__ == "__main__":
    main()
