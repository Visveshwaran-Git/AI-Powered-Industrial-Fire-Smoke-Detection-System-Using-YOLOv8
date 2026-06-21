import cv2
import threading
import time
import os
import numpy as np
import random

class VideoStreamingManager:
    def __init__(self):
        self.camera_source = "simulation"  # "simulation", 0 (webcam), or file path
        self.cap = None
        self.is_running = False
        self.frame = None
        self.thread = None
        self.lock = threading.Lock()
        
        # Simulation parameters
        self.gauge_angle = 0
        self.smoke_particles = []
        self.fire_flames = []
        
    def start(self, source="simulation"):
        """Starts the video capture thread with the given source."""
        self.stop()
        
        # Convert numeric string to int if it represents an index
        if isinstance(source, str) and source.isdigit():
            self.camera_source = int(source)
        else:
            self.camera_source = source
            
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"Video capture thread started with source: {self.camera_source}")
        
    def stop(self):
        """Stops the video capture thread and releases camera resources."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.5)
            self.thread = None
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.frame = None
        print("Video capture thread stopped.")
            
    def get_frame(self):
        """Returns the current frame in a thread-safe manner."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
            
    def _capture_loop(self):
        """Main loop that grabs frames from the camera, file, or generates simulation."""
        
        # If not simulating, try to initialize cv2.VideoCapture
        if self.camera_source != "simulation":
            try:
                self.cap = cv2.VideoCapture(self.camera_source)
                # Set resolution to 640x480 for consistent YOLO inference performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Check if camera opened successfully
                if not self.cap.isOpened():
                    print(f"Warning: Could not open video source {self.camera_source}. Falling back to simulation.")
                    self.camera_source = "simulation"
            except Exception as e:
                print(f"Error opening video capture: {e}. Falling back to simulation.")
                self.camera_source = "simulation"

        last_frame_time = time.time()
        fps_target = 24.0
        frame_delay = 1.0 / fps_target
        
        while self.is_running:
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
                continue
            last_frame_time = time.time()
            
            if self.camera_source == "simulation":
                # Generate a beautiful synthetic industrial dashboard feed
                frame = self._generate_simulated_frame()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    # If it's a video file, loop it
                    if isinstance(self.camera_source, str) and os.path.exists(self.camera_source):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            frame = self._generate_simulated_frame()
                    else:
                        # Webcam disconnected or error
                        print("Webcam read failed. Reconnecting or falling back to simulation...")
                        time.sleep(1.0)
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.camera_source)
                        continue
            
            # Update frame thread-safely
            with self.lock:
                self.frame = frame

        # Clean up at exit
        if self.cap:
            self.cap.release()
            self.cap = None

    def _generate_simulated_frame(self):
        """Generates a 640x480 frame simulating an industrial boiler room scene."""
        width, height = 640, 480
        # Create dark industrial background (deep grey gradient-ish)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (20, 22, 25) # Dark graphite background
        
        # Draw background grids
        grid_size = 40
        for x in range(0, width, grid_size):
            cv2.line(frame, (x, 0), (x, height), (32, 34, 38), 1)
        for y in range(0, height, grid_size):
            cv2.line(frame, (0, y), (width, y), (32, 34, 38), 1)
            
        # Draw metallic structural pillars
        cv2.rectangle(frame, (0, 0), (40, height), (40, 43, 48), -1)
        cv2.rectangle(frame, (width-40, 0), (width, height), (40, 43, 48), -1)
        # Pillar bolts
        for y in range(40, height, 80):
            cv2.circle(frame, (20, y), 5, (80, 85, 90), -1)
            cv2.circle(frame, (width-20, y), 5, (80, 85, 90), -1)
            
        # Draw horizontal steam pipes near the top
        # Large steel pipe
        cv2.rectangle(frame, (40, 50), (width-40, 90), (100, 105, 110), -1)
        cv2.rectangle(frame, (40, 53), (width-40, 60), (160, 165, 170), -1) # Highlight
        cv2.rectangle(frame, (40, 80), (width-40, 88), (60, 65, 70), -1)   # Shadow
        
        # Steam Valve Wheel
        cv2.circle(frame, (120, 70), 20, (30, 30, 200), 4) # Red valve wheel
        cv2.circle(frame, (120, 70), 5, (80, 85, 90), -1)
        cv2.line(frame, (120, 50), (120, 90), (30, 30, 200), 3)
        cv2.line(frame, (100, 70), (140, 70), (30, 30, 200), 3)

        # Draw a heavy industrial boiler tank in the center
        # Tank body
        cv2.rectangle(frame, (200, 180), (440, 450), (60, 63, 68), -1)
        cv2.rectangle(frame, (200, 180), (440, 200), (80, 85, 90), -1) # Tank lid/cap
        # Highlight on tank
        cv2.rectangle(frame, (210, 200), (230, 450), (90, 95, 100), -1)
        
        # Tank warning decal
        # Yellow and black caution stripes
        decal_y = 230
        for i in range(0, 80, 16):
            cv2.rectangle(frame, (280 + i, decal_y), (280 + i + 8, decal_y + 15), (0, 200, 230), -1) # Yellow BGR
            cv2.rectangle(frame, (280 + i + 8, decal_y), (280 + i + 16, decal_y + 15), (20, 20, 20), -1) # Black BGR
            
        # Draw a pressure gauge dial
        dial_center = (320, 320)
        cv2.circle(frame, dial_center, 40, (230, 230, 230), -1) # Dial face
        cv2.circle(frame, dial_center, 42, (100, 105, 110), 3)   # Bezel
        # Tick marks
        for angle in range(0, 360, 30):
            rad = np.deg2rad(angle)
            p1 = (int(dial_center[0] + 32 * np.cos(rad)), int(dial_center[1] + 32 * np.sin(rad)))
            p2 = (int(dial_center[0] + 38 * np.cos(rad)), int(dial_center[1] + 38 * np.sin(rad)))
            cv2.line(frame, p1, p2, (40, 40, 40), 2)
            
        # Animate the gauge needle slightly
        self.gauge_angle = (self.gauge_angle + 3) % 360
        needle_angle = 180 + 60 * np.sin(np.deg2rad(self.gauge_angle))
        n_rad = np.deg2rad(needle_angle)
        needle_end = (int(dial_center[0] + 32 * np.cos(n_rad)), int(dial_center[1] + 32 * np.sin(n_rad)))
        cv2.line(frame, dial_center, needle_end, (0, 0, 220), 2) # Red needle
        cv2.circle(frame, dial_center, 5, (20, 20, 20), -1)
        
        # Text Label on the boiler
        cv2.putText(frame, "TEMP DANGER LEVEL", (250, 385), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "PSI MONITOR", (278, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)

        # Simulation cycle changes state every 6 seconds to show fire and smoke
        epoch_seconds = int(time.time())
        seed_cycle = (epoch_seconds // 6) % 5
        
        # Draw physical fire/smoke graphics on top of the boiler room if in those cycles
        if seed_cycle in [1, 3]:  # Fire cycles
            self._draw_fire_simulation(frame, 320, 175)
        if seed_cycle in [2, 3]:  # Smoke cycles
            self._draw_smoke_simulation(frame, 320, 160)
            
        # Draw Camera overlay info (text, timestamp)
        cv2.putText(frame, "CAM-01 | HEAVY BOILER ROOM", (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Red recording dot flashing
        if epoch_seconds % 2 == 0:
            cv2.circle(frame, (590, 30), 6, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (605, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp_str, (50, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 165, 170), 1, cv2.LINE_AA)
        
        return frame
        
    def _draw_fire_simulation(self, frame, x_base, y_base):
        """Draws dynamic, realistic flickering fire polygon elements on the frame."""
        t = time.time()
        # Create shifting flame polygons
        for i in range(3):
            # Scale coordinates slightly for multiple layers (inner core is yellow, outer is orange/red)
            flicker_x = int(12 * np.sin(t * 15 + i * 2))
            flicker_y = int(8 * np.cos(t * 12 + i))
            
            w_scale = 1.0 - (i * 0.25)
            h_scale = 1.0 - (i * 0.20)
            
            pts = np.array([
                [x_base - int(40*w_scale) + flicker_x // 2, y_base],
                [x_base - int(20*w_scale) + flicker_x, y_base - int(35*h_scale) + flicker_y],
                [x_base + flicker_x // 3, y_base - int(70*h_scale) + flicker_y // 2],
                [x_base + int(15*w_scale) + flicker_x, y_base - int(45*h_scale) + flicker_y],
                [x_base + int(35*w_scale) + flicker_x // 2, y_base]
            ], np.int32)
            
            # Fire Colors (BGR): 
            # Layer 0: Dark orange-red
            # Layer 1: Bright safety orange
            # Layer 2: Yellow-white core
            colors = [
                (0, 69, 255),    # Red-Orange BGR
                (0, 140, 255),   # Orange BGR
                (0, 230, 255)    # Yellow BGR
            ]
            
            # Draw filled flame layer
            cv2.fillPoly(frame, [pts], colors[i])
            
    def _draw_smoke_simulation(self, frame, x_base, y_base):
        """Draws semi-transparent rising smoke puff circles."""
        t = time.time()
        # Initialize or update smoke puffs
        # Each particle is a list: [x, y, radius, opacity, speed]
        # Generate new particle occasionally
        if len(self.smoke_particles) < 15 and int(t * 10) % 3 == 0:
            self.smoke_particles.append([
                x_base + random.randint(-15, 15), 
                y_base, 
                random.randint(12, 22), 
                0.6, # Max Opacity
                random.uniform(1.8, 3.2) # Speed
            ])
            
        active_particles = []
        # Overlay sheet to draw transparent smoke
        overlay = frame.copy()
        
        for p in self.smoke_particles:
            # Move particle up
            p[1] -= p[4] # Decrease y
            # Expand radius slightly as it rises
            p[2] += 0.4
            # Fade out
            p[3] -= 0.012
            
            if p[1] > 30 and p[3] > 0:
                # Add horizontal drift using sine wave
                drift_x = int(p[0] + 25 * np.sin(p[1] / 30.0 + t))
                
                # Draw puff circle
                cv2.circle(overlay, (drift_x, int(p[1])), int(p[2]), (130, 135, 140), -1)
                active_particles.append(p)
                
        self.smoke_particles = active_particles
        
        # Apply alpha blend
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
