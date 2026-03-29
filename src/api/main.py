from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import cv2
import os

app = FastAPI(title="CogniVision Dashboard")

# Mount static files and templates
# app.mount("/static", StaticFiles(directory="src/api/static"), name="static")
# templates = Jinja2Templates(directory="src/api/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <html>
        <head>
            <title>CogniVision</title>
            <style>
                body { font-family: sans-serif; background: #0f172a; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                .container { text-align: center; background: rgba(255, 255, 255, 0.05); padding: 2rem; border-radius: 1rem; backdrop-filter: blur(10px); }
                h1 { margin-bottom: 1rem; }
                .video-feed { width: 640px; height: 480px; background: #000; border-radius: 0.5rem; margin-bottom: 1rem; }
                .stats { font-size: 1.2rem; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>CogniVision Dashboard 👁️</h1>
                <div class="video-feed">
                    <!-- Video feed will go here -->
                    <p style="padding-top: 200px; color: #64748b;">[ Video Feed Loading... ]</p>
                </div>
                <div class="stats">
                    <p>Attention Score: --%</p>
                    <p>Status: Initializing AI Core...</p>
                </div>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
