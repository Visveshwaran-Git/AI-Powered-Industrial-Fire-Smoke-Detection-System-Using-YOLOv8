import sqlite3
import os
from datetime import datetime

# Resolve paths relative to the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DB_DIR, 'detections.db')
SNAPSHOTS_DIR = os.path.join(DB_DIR, 'snapshots')

def init_db():
    """Initializes the database and creates directories if they do not exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            camera_id TEXT NOT NULL,
            snapshot_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_detection(class_name: str, confidence: float, camera_id: str, snapshot_path: str = None):
    """Inserts a detection event into the database."""
    init_db()  # Safety check
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO detections (timestamp, class_name, confidence, camera_id, snapshot_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, class_name, confidence, camera_id, snapshot_path))
    conn.commit()
    conn.close()
    return timestamp

def get_recent_detections(limit: int = 50):
    """Fetches the most recent detection logs."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, class_name, confidence, camera_id, snapshot_path
        FROM detections
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_detection_stats():
    """Calculates statistics for the dashboard dashboard panels."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Total counts
    cursor.execute("SELECT COUNT(*) as count FROM detections")
    total = cursor.fetchone()['count']
    
    # Counts by class
    cursor.execute("SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name")
    by_class = {row['class_name']: row['count'] for row in cursor.fetchall()}
    
    # Average confidence
    cursor.execute("SELECT AVG(confidence) as avg_conf FROM detections")
    avg_conf = cursor.fetchone()['avg_conf']
    avg_conf = round(avg_conf, 2) if avg_conf else 0.0
    
    # Recent trend: Detections grouped by hour (for the line graph)
    # Get detections from the last 24 hours
    cursor.execute('''
        SELECT strftime('%H:00', timestamp) as hour, COUNT(*) as count 
        FROM detections 
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY hour
        ORDER BY hour ASC
    ''')
    hourly_trends = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "total_detections": total,
        "fire_count": by_class.get("fire", 0),
        "smoke_count": by_class.get("smoke", 0),
        "average_confidence": avg_conf,
        "hourly_trends": hourly_trends
    }

def clear_logs():
    """Deletes all logged detections and resets the auto-increment ID."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='detections'")
    conn.commit()
    conn.close()
    
    # Delete saved snapshots
    if os.path.exists(SNAPSHOTS_DIR):
        for file in os.listdir(SNAPSHOTS_DIR):
            file_path = os.path.join(SNAPSHOTS_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
