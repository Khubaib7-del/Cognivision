#!/usr/bin/env python3
"""
Simple headless webcam test for CogniVision pipeline.
Runs for a specified duration and prints per-frame stats.
Use --display to show OpenCV windows (not recommended in headless environments).
"""
import time
import argparse
from pathlib import Path
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.core.pipeline import ProcessingPipeline
from src.core.scorer import CogniVisionScorer
from src.config import DETECTOR_MODEL


def main(duration=10, display=False, detector_model=DETECTOR_MODEL):
    pipeline = ProcessingPipeline(detector_model=detector_model)
    scorer = CogniVisionScorer(phone_penalty=20)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠ Cannot open webcam (index 0). Exiting.")
        return 2

    start = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Failed to read frame from webcam")
                break

            detections = pipeline.process_frame(frame)
            score = scorer.calculate_class_score(detections)
            report = scorer.get_individual_report(detections)

            # Print concise per-frame stats
            print(f"Frame {frame_idx}: detections={len(detections)}, score={score}, report={report}")

            if display:
                # Overlay boxes and labels
                for d in detections:
                    x1, y1, x2, y2 = [int(v) for v in d['bbox']]
                    color = (0, 255, 0) if d['status'] == 'attentive' else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{d['status']}:{d.get('confidence',0):.2f}"
                    cv2.putText(frame, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                cv2.imshow('CogniVision Webcam Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if time.time() - start > duration:
                break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    print("Webcam test complete.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=10, help='Duration in seconds')
    parser.add_argument('--display', action='store_true', help='Show display windows')
    parser.add_argument('--model', type=str, default=DETECTOR_MODEL, help='Detector model file')
    args = parser.parse_args()
    raise SystemExit(main(duration=args.duration, display=args.display, detector_model=args.model))
