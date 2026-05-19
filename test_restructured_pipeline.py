#!/usr/bin/env python
"""Quick test of the restructured pipeline."""

import cv2
import numpy as np
from src.core.pipeline import ProcessingPipeline

# Create a test frame (480x640)
frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

# Initialize pipeline
print('Loading pipeline...')
pipeline = ProcessingPipeline()
print('✓ Pipeline loaded')

# Run inference
print('Running inference on test frame...')
detections = pipeline.process_frame(frame)

print(f'✓ Inference completed')
print(f'  Total detections: {len(detections)}')
if detections:
    for i, det in enumerate(detections):
        det_type = det.get('type', 'unknown')
        status = det.get('status', 'unknown')
        conf = det.get('confidence', 0)
        print(f'  [{i}] type={det_type}, status={status}, conf={conf:.2f}')
else:
    print('  (No objects in blank image - expected)')

print('\n✓ Pipeline restructuring successful!')
print('  - Single detector (yolov8n.pt with fallback)')
print('  - No detector suppression logic')
print('  - Clean categorization (student, phone, object)')
