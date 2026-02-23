import os
import cv2
import threading
import time
import queue
import json
from datetime import datetime
from flask import Flask, render_template, Response, send_from_directory, request, jsonify

from storage_manager import StorageManager
from vision_tracker import VisionTracker
from violation_engine import ViolationEngine

app = Flask(__name__)

# Change this to 0 for a local webcam, or your RTSP link for an IP camera
VIDEO_PATH = 'Riyadh metro 1.MOV' 
FPS = 30

violation_queue = queue.Queue()
clients = []
clients_lock = threading.Lock()
def handle_new_violation(violation_data):
    """Callback triggered by StorageManager. Broadcasts to ALL open tabs."""
    with clients_lock:
        for client_queue in clients:
            client_queue.put(violation_data)
storage = StorageManager(on_new_violation_callback=handle_new_violation)
tracker = VisionTracker(fps=FPS)
engine = ViolationEngine(storage)

# Global states for our producer-consumer pipeline
raw_frame = None
processed_frame_bytes = None
lock = threading.Lock()

def camera_reader():
    """PRODUCER: Reads frames from the camera. Throttled to simulate real-time."""
    global raw_frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_delay = 1.0 / FPS
    
    while cap.isOpened():
        start_time = time.time()
        
        success, frame = cap.read()
        if success:
            with lock:
                raw_frame = frame
        else:
            # The video ended! Loop it and RESET our databases for testing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            storage.saved_snaps.clear()  # Allow violations to trigger again
            tracker.uuid_map.clear()     # Reset vehicle UUIDs
            continue
        
        read_time = time.time() - start_time
        time_to_sleep = frame_delay - read_time
        if time_to_sleep > 0:
            time.sleep(time_to_sleep - (time_to_sleep * 0.3))

def ai_processor():
    """CONSUMER: Grabs the freshest frame, runs AI, and drops skipped frames."""
    global raw_frame, processed_frame_bytes
    
    while True:
        with lock:
            # Copy the frame so the camera reader can keep updating the original
            frame = raw_frame.copy() if raw_frame is not None else None
            
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Use actual system time. This is critical for live cameras so your 
        # VisionTracker speed calculation (Distance / Time) remains accurate!
        current_time = time.time()
        
        tracked_objects = tracker.track_and_get_speeds(frame, current_time)
        processed = engine.process_and_draw(frame, tracked_objects, current_time)
        
        ret, buffer = cv2.imencode('.jpg', processed)
        if ret:
            with lock:
                processed_frame_bytes = buffer.tobytes()

# Start the decoupled threads
threading.Thread(target=camera_reader, daemon=True).start()
threading.Thread(target=ai_processor, daemon=True).start()

def generate_frames():
    """STREAMER: Yields the latest processed frame instantly to the web browser."""
    last_sent = None
    while True:
        with lock:
            frame = processed_frame_bytes
            
        if frame is None or frame == last_sent:
            time.sleep(0.005)
            continue
            
        last_sent = frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_violations')
def stream_violations():
    """Pushes new violations to the browser in real-time."""
    def event_stream():
        client_queue = queue.Queue()
        with clients_lock:
            clients.append(client_queue)
            
        try:
            while True:
                try:
                    # Wait up to 2 seconds for a violation
                    violation = client_queue.get(timeout=2.0) 
                    yield f"data: {json.dumps(violation)}\n\n"
                except queue.Empty:
                    # Send a hidden comment to keep the browser connection alive
                    yield ": keep-alive\n\n"
        finally:
            with clients_lock:
                if client_queue in clients:
                    clients.remove(client_queue)

    return Response(event_stream(), mimetype="text/event-stream")

def get_all_violations():
    """Reads all violation records and sorts them strictly by the Timestamp key in the txt info."""
    violations_list = []
    if os.path.exists(storage.IMG_DIR):
        for filename in os.listdir(storage.IMG_DIR):
            if filename.endswith('.txt'):
                txt_path = os.path.join(storage.IMG_DIR, filename)
                info = {'base_filename': filename.replace('.txt', '')}
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        for line in f.readlines():
                            if ": " in line:
                                key, val = line.split(": ", 1)
                                info[key.strip()] = val.strip()
                    violations_list.append(info)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    
    # --- SORT BY THE 'Timestamp' KEY FROM THE TXT FILE ---
    def sort_by_txt_timestamp(v):
        time_str = v.get('Timestamp', '')
        try:
            # Matches the format saved in storage_manager.py: "%Y-%m-%d %I:%M:%S %p"
            
            return datetime.strptime(time_str, "%Y-%m-%d %I:%M:%S %p")
        except ValueError:
            # If the timestamp is missing or corrupted, push this record to the bottom
            return datetime.min

    # Sort descending (newest first)
    violations_list.sort(key=sort_by_txt_timestamp, reverse=True)
    return violations_list

@app.route('/delete_violations', methods=['POST'])
def delete_violations():
    """Receives a list of filenames from the frontend and deletes the .txt and .jpg files."""
    data = request.get_json()
    filenames = data.get('filenames', [])
    
    deleted_count = 0
    for base_name in filenames:
        txt_path = os.path.join(storage.IMG_DIR, f"{base_name}.txt")
        img_path = os.path.join(storage.IMG_DIR, f"{base_name}.jpg")
        
        try:
            if os.path.exists(txt_path):
                os.remove(txt_path)
            if os.path.exists(img_path):
                os.remove(img_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {base_name}: {e}")
            
    return jsonify({"success": True, "deleted": deleted_count})
@app.route('/violation/<base_filename>')
def violation_detail(base_filename):
    txt_path = os.path.join(storage.IMG_DIR, f"{base_filename}.txt")
    info = {}
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if ": " in line:
                    key, val = line.split(": ", 1)
                    info[key.strip()] = val.strip()
    except FileNotFoundError:
        return "Violation details not found.", 404
        
    return render_template('violation.html', info=info, img_file=f"{base_filename}.jpg")
@app.route('/database')
def database():
    return render_template('database.html', violations=get_all_violations())
@app.route('/violations/images/<filename>')
def serve_image(filename):
    return send_from_directory(storage.IMG_DIR, filename)
@app.route('/laner')
def laner():
    """Serves the Lane Builder tool."""
    return render_template('laner.html')
@app.route('/api/lanes', methods=['GET', 'POST'])
def manage_lanes():
    """API to load and save lane configurations to JSON."""
    config_file = "lanes_config.json"
    
    if request.method == 'POST':
        # Web UI is saving new lanes
        data = request.get_json()
        lanes = data.get('lanes', [])
        
        with open(config_file, 'w') as f:
            json.dump(lanes, f)
            
        # THE MAGIC: Tell the engine to reload the file instantly!
        engine.load_lanes()
        return jsonify({"success": True, "message": "Lanes saved and applied dynamically."})
        
    else:
        # Web UI is requesting the current lanes to display them
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return jsonify({"lanes": json.load(f)})
            except Exception:
                pass
        return jsonify({"lanes": []})
@app.route('/api/reference_frame')
def reference_frame():
    """Grabs a single frame from the video to use as the drawing background."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, frame = cap.read()
    cap.release()
    
    if success:
        ret, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "Failed to load video", 500
if __name__ == "__main__":
    # We turn off the reloader to prevent the background threads from starting twice
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)