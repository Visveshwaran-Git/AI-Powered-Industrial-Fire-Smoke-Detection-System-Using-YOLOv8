import os
import cv2
import numpy as np
import time
import random
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

class FireSmokeDetector:
    def __init__(self, confidence_threshold=0.25):
        self.conf_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False
        self.class_names = {}
        self.use_simulation = False
        self.load_model()
        
    def load_model(self):
        """Attempts to load the custom YOLOv8 fire/smoke model. Fallbacks to standard or mock if offline."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        model_path = os.path.join(data_dir, 'best.pt')
        
        try:
            print(f"Loading custom fire/smoke model weights from: {model_path}")
            if not os.path.exists(model_path):
                print("Weights not found locally. Downloading from Hugging Face Hub (rabahdev/fire-smoke-yolov8n)...")
                try:
                    # Download using Hugging Face API
                    downloaded_path = hf_hub_download(
                        repo_id="rabahdev/fire-smoke-yolov8n", 
                        filename="best.pt",
                        local_dir=data_dir,
                        local_dir_use_symlinks=False
                    )
                    # Handle cases where local_dir is ignored or returns path in cache
                    if os.path.exists(downloaded_path) and downloaded_path != model_path:
                        import shutil
                        shutil.copy2(downloaded_path, model_path)
                    print("Weights downloaded successfully via Hugging Face Hub.")
                except Exception as e:
                    print(f"Hugging Face download failed: {e}. Trying direct HTTP download...")
                    import urllib.request
                    url = "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                    print("Direct HTTP download completed.")
            
            # Load Ultralytics YOLOv8
            self.model = YOLO(model_path)
            self.model_loaded = True
            # Normalize class names to lowercase
            self.class_names = {int(k): v.lower() for k, v in self.model.names.items()}
            print("Successfully loaded YOLOv8 model. Classes:", self.class_names)
        except Exception as e:
            print(f"Could not load custom YOLOv8 model: {e}")
            print("Trying to fall back to standard COCO YOLOv8n model...")
            try:
                self.model = YOLO("yolov8n.pt")
                self.model_loaded = True
                self.class_names = {int(k): v.lower() for k, v in self.model.names.items()}
                print("Standard YOLOv8n loaded. Since this is COCO, fire/smoke will be simulated.")
                self.use_simulation = True
            except Exception as ex:
                print(f"Critical error: YOLO library or standard model failed to load: {ex}")
                print("Entering full simulation mode (no deep learning runtime required).")
                self.model = None
                self.model_loaded = False
                self.use_simulation = True

    def predict(self, frame, conf_override=None):
        """
        Runs inference on the frame.
        Returns:
            detections: List of dicts, e.g., [{"class": "fire", "confidence": 0.85, "box": [x1, y1, x2, y2]}]
            annotated_frame: Frame with drawn bounding boxes and labels
        """
        conf_thr = conf_override if conf_override is not None else self.conf_threshold
        
        # If in full simulation mode or fallback is active
        if self.use_simulation or not self.model_loaded or self.model is None:
            return self._run_simulation(frame, conf_thr)
            
        try:
            results = self.model(frame, verbose=False, conf=conf_thr)[0]
            detections = []
            annotated_frame = frame.copy()
            
            # BGR Color settings
            colors = {
                "fire": (0, 107, 255),    # Safety Orange
                "smoke": (128, 128, 128)  # Slate Grey
            }
            fallback_color = (0, 200, 200) # Yellow-green
            
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(int, coords)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                
                # Standardize category
                mapped_name = "fire" if "fire" in class_name else ("smoke" if "smoke" in class_name else class_name)
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class": mapped_name,
                    "confidence": round(conf, 2)
                })
                
                # Select box color
                color = colors.get(mapped_name, fallback_color)
                
                # Draw thick rounded rectangle borders
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw high-visibility label banner
                label = f"{mapped_name.upper()} {conf*100:.1f}%"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            
            return detections, annotated_frame
            
        except Exception as e:
            print(f"Error during model inference: {e}. Falling back to simulation.")
            return self._run_simulation(frame, conf_thr)

    def _run_simulation(self, frame, conf_thr):
        """Simulates detections for demo/testing purposes when YOLO is not available or failing."""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        detections = []
        
        # We simulate a detection randomly every few seconds or if there's significant brightness
        # For a cooler effect, let's use the current time to create stable simulated bounding boxes
        # so they don't flicker randomly on every single frame, but rather stay for a few seconds.
        epoch_seconds = int(time.time())
        seed_cycle = (epoch_seconds // 6) % 5  # Changes state every 6 seconds
        
        # Seed 0: no detections
        # Seed 1: Fire detection
        # Seed 2: Smoke detection
        # Seed 3: Both fire and smoke
        # Seed 4: no detections
        
        colors = {
            "fire": (0, 107, 255),    # Safety Orange BGR
            "smoke": (120, 120, 120)  # Slate Grey
        }
        
        sim_detections_to_make = []
        if seed_cycle == 1:
            sim_detections_to_make.append(("fire", 0.88, [int(w*0.35), int(h*0.4), int(w*0.65), int(h*0.8)]))
        elif seed_cycle == 2:
            sim_detections_to_make.append(("smoke", 0.79, [int(w*0.2), int(h*0.15), int(w*0.8), int(h*0.65)]))
        elif seed_cycle == 3:
            sim_detections_to_make.append(("smoke", 0.72, [int(w*0.15), int(h*0.1), int(w*0.85), int(h*0.55)]))
            sim_detections_to_make.append(("fire", 0.92, [int(w*0.4), int(h*0.5), int(w*0.6), int(h*0.85)]))
            
        for name, conf, box in sim_detections_to_make:
            if conf >= conf_thr:
                x1, y1, x2, y2 = box
                detections.append({
                    "box": box,
                    "class": name,
                    "confidence": conf
                })
                
                # Draw bounding box
                color = colors[name]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"{name.upper()} (SIM) {conf*100:.1f}%"
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + txt_w + 10, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                            
                # Optional: draw visual visual effects on the mock frame to make it look active
                # Overlay a light colored transparent rect over the box to indicate region of interest
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.12, annotated_frame, 0.88, 0, annotated_frame)
                
        return detections, annotated_frame
