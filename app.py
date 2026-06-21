import os
import time
import shutil
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import custom modules
from database import init_db, log_detection, get_recent_detections, get_detection_stats, clear_logs, DB_DIR, SNAPSHOTS_DIR
from detector import FireSmokeDetector
from video_stream import VideoStreamingManager

# Create directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
PROCESSED_DIR = os.path.join(STATIC_DIR, 'processed')
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# Initialize database
init_db()

# Initialize Detector and Streaming Manager
detector = FireSmokeDetector(confidence_threshold=0.25)
streamer = VideoStreamingManager()

# Start simulation stream by default on startup
streamer.start(source="simulation")

app = FastAPI(title="Industrial Fire & Smoke Detection System", version="1.0")

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Throttling dictionary to prevent DB flooding (only log the same event type every 5 seconds)
last_logged_time = {"fire": 0.0, "smoke": 0.0}
alerts_active = {"fire": False, "smoke": False}
latest_alert_info = {"fire": None, "smoke": None}

def stream_generator():
    """Generates JPEG frames for real-time video stream (MJPEG format)."""
    global last_logged_time, alerts_active, latest_alert_info
    
    frame_count = 0
    while True:
        frame = streamer.get_frame()
        if frame is None:
            # If frame is not available yet, send a blank slate placeholder
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "No Video Source Active", (170, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _, jpeg = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        frame_count += 1
        # Run YOLOv8 detection
        detections, annotated_frame = detector.predict(frame)
        
        now_time = time.time()
        current_frame_has = {"fire": False, "smoke": False}
        
        for det in detections:
            cls = det["class"]
            conf = det["confidence"]
            
            if cls in ["fire", "smoke"]:
                current_frame_has[cls] = True
                
                # Check throttle limits
                if now_time - last_logged_time[cls] > 5.0:
                    last_logged_time[cls] = now_time
                    
                    # Capture and save a snapshot
                    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    snapshot_filename = f"snap_{cls}_{timestamp_slug}.jpg"
                    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
                    cv2.imwrite(snapshot_path, annotated_frame)
                    
                    # Log event in SQLite
                    db_time = log_detection(
                        class_name=cls,
                        confidence=conf,
                        camera_id="CAM-01",
                        snapshot_path=f"/snapshots/{snapshot_filename}"
                    )
                    
                    # Set current alert info
                    latest_alert_info[cls] = {
                        "class_name": cls,
                        "confidence": conf,
                        "time": db_time,
                        "snapshot": f"/snapshots/{snapshot_filename}"
                    }
        
        # Update alerts state
        alerts_active["fire"] = current_frame_has["fire"]
        alerts_active["smoke"] = current_frame_has["smoke"]

        # Encode and stream frame
        _, jpeg = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        # Limit processing frame rate to match streamer (~24 FPS) to reduce CPU load
        time.sleep(0.035)

@app.get("/")
def get_root():
    """Serves the main dashboard user interface."""
    index_file = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(status_code=404, content={"message": "Frontend index.html not found. Please verify folder structure."})

@app.get("/api/stream")
def get_video_stream():
    """Returns the live YOLOv8-annotated MJPEG camera stream."""
    return StreamingResponse(stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
def get_stats():
    """Returns real-time detection counts, system logs summary, and alert status."""
    stats = get_detection_stats()
    stats.update({
        "alerts_active": alerts_active,
        "latest_alert_info": latest_alert_info,
        "camera_source": str(streamer.camera_source),
        "detector_loaded": detector.model_loaded,
        "model_name": "YOLOv8n-Fire-Smoke" if not detector.use_simulation else "YOLOv8n-Simulation"
    })
    return stats

@app.get("/api/history")
def get_history(limit: int = 50):
    """Retrieves the history of detections from the SQLite database."""
    return get_recent_detections(limit=limit)

@app.post("/api/clear-history")
def post_clear_history():
    """Clears SQLite history and removes physical snapshots."""
    global latest_alert_info, alerts_active
    clear_logs()
    latest_alert_info = {"fire": None, "smoke": None}
    alerts_active = {"fire": False, "smoke": False}
    return {"status": "success", "message": "Detection history database and snapshots cleared."}

@app.get("/api/config")
def get_config():
    """Returns the active detector configuration."""
    return {
        "confidence_threshold": detector.conf_threshold,
        "camera_source": str(streamer.camera_source),
        "use_simulation": detector.use_simulation
    }

@app.post("/api/config")
def update_config(data: dict):
    """Updates the detection confidence threshold or camera stream source in real-time."""
    # Handle confidence threshold update
    if "confidence_threshold" in data:
        try:
            val = float(data["confidence_threshold"])
            if 0.0 <= val <= 1.0:
                detector.conf_threshold = val
                print(f"Config update: Confidence threshold set to {val}")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid confidence threshold value")
            
    # Handle camera source update
    if "camera_source" in data:
        new_source = data["camera_source"]
        # Try numeric string (e.g. "0" for webcam)
        if new_source.isdigit():
            new_source = int(new_source)
        streamer.start(source=new_source)
        print(f"Config update: Switched camera source to {new_source}")
        
    return {"status": "success", "config": get_config()}

@app.post("/api/upload")
async def handle_upload(
    file: UploadFile = File(...),
    conf: float = Form(0.25)
):
    """Handles static image or video upload, runs YOLOv8, saves output, and returns detection results."""
    # Ensure processed files folder exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    temp_input_path = os.path.join(PROCESSED_DIR, f"temp_{unique_filename}")
    output_filename = f"proc_{unique_filename}"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    
    # Save the uploaded file temporarily
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Check if image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    detections_summary = []
    
    if file_extension in image_extensions:
        # Process Image
        img = cv2.imread(temp_input_path)
        if img is None:
            os.unlink(temp_input_path)
            raise HTTPException(status_code=400, detail="Invalid image file uploaded.")
            
        detections, annotated_img = detector.predict(img, conf_override=conf)
        cv2.imwrite(output_path, annotated_img)
        
        # Log to DB if detections found
        for d in detections:
            log_detection(
                class_name=d["class"],
                confidence=d["confidence"],
                camera_id="UPLOADED_IMAGE",
                snapshot_path=f"/static/processed/{output_filename}"
            )
            detections_summary.append({
                "class": d["class"],
                "confidence": d["confidence"]
            })
            
        # Clean up temp file
        os.unlink(temp_input_path)
        
        return {
            "media_type": "image",
            "processed_url": f"/static/processed/{output_filename}",
            "detections": detections_summary
        }
        
    elif file_extension in video_extensions:
        # Process Video
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            os.unlink(temp_input_path)
            raise HTTPException(status_code=400, detail="Could not open video file.")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare video writer (h264 or mp4v encoding)
        # Using H264 (avc1) or mp4v for high browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detected_classes = set()
        max_confidences = {}
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every frame, or skip frames if needed. For short upload videos, we process all frames
            detections, annotated_frame = detector.predict(frame, conf_override=conf)
            out.write(annotated_frame)
            
            # Record detections
            for d in detections:
                cls = d["class"]
                cf = d["confidence"]
                detected_classes.add(cls)
                if cls not in max_confidences or cf > max_confidences[cls]:
                    max_confidences[cls] = cf
            
            frame_idx += 1
            # Cap video processing length at 30 seconds to prevent denial of service (server freezes)
            if frame_idx > fps * 30:
                print("Video upload processing capped at 30 seconds.")
                break
                
        cap.release()
        out.release()
        
        # Log detected elements to database
        for cls in detected_classes:
            log_detection(
                class_name=cls,
                confidence=max_confidences[cls],
                camera_id="UPLOADED_VIDEO",
                snapshot_path=f"/static/processed/{output_filename}"
            )
            detections_summary.append({
                "class": cls,
                "confidence": max_confidences[cls]
            })
            
        # Clean up temp file
        os.unlink(temp_input_path)
        
        return {
            "media_type": "video",
            "processed_url": f"/static/processed/{output_filename}",
            "detections": detections_summary
        }
    else:
        os.unlink(temp_input_path)
        raise HTTPException(status_code=400, detail="Unsupported file format.")

# Mount the static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Mount snapshots directory
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    print("Launching FastAPI Web Server for Fire & Smoke Detection System...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
