PHASE: Deployment quickstart

1) Build and run with Docker (recommended):

```bash
# From project root
docker build -t cognivision:latest .
docker run -p 8001:8001 -v "$(pwd)/data:/app/data:ro" cognivision:latest
```

Or use docker-compose:

```bash
docker-compose up --build
```

2) API endpoints:
- `/` -> Dashboard (MJPEG feed embedded)
- `/video_feed` -> MJPEG stream
- `/api/infer` -> POST image (multipart) -> returns detections + class_score

3) Notes:
- For laptop/desktop webcam access, run locally (not in container) or map devices into container (platform-dependent).
- Large model files should remain on host and be mounted into `/app/data/models`.
- For production, switch to `opencv-python-headless` to reduce image size and avoid GUI deps.
