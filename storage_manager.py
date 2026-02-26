import os
import cv2
from datetime import datetime

class StorageManager:
    # We add an optional callback function to notify the Flask app of new violations
    def __init__(self, on_new_violation_callback=None):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        self.SAVE_DIR = os.path.join(self.BASE_DIR, "violations")
        self.IMG_DIR = os.path.join(self.SAVE_DIR, "images")
        
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.IMG_DIR, exist_ok=True)
        
        self.saved_snaps = set()
        self.on_new_violation = on_new_violation_callback

    def save_snapshot(self, frame, track_id, violation_type, lane_num=None, bbox=None):
        key = (str(track_id), str(violation_type))
        if key in self.saved_snaps:
            return
        self.saved_snaps.add(key)

        now_dt = datetime.now()
        ts_str = now_dt.strftime("%Y-%m-%d %I:%M:%S %p")
        overlay = frame.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)

        text = f"{violation_type} | ID:{track_id} | {ts_str}"
        if lane_num is not None:
            text += f" | L{lane_num}"

        cv2.putText(overlay, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        safe_name = violation_type.replace("/", "-").replace(" ", "_")
        base_filename = f"{safe_name}_id{track_id}_{now_dt.strftime('%Y%m%d_%H%M%S')}"
        
        img_path = os.path.join(self.IMG_DIR, f"{base_filename}.jpg")
        txt_path = os.path.join(self.IMG_DIR, f"{base_filename}.txt")

        cv2.imwrite(img_path, overlay)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Violation Type: {violation_type}\n")
            f.write(f"Vehicle ID: {track_id}\n")
            f.write(f"Lane: {lane_num}\n")
            f.write(f"Timestamp: {ts_str}\n")
            f.write(f"Image File: {base_filename}.jpg\n")
            
        print(f"[ALERT] Saved: {violation_type} by Vehicle ID {track_id}")

        if self.on_new_violation:
            violation_data = {
                "id": track_id,
                "type": violation_type,
                "timestamp": ts_str,
                "base_filename": base_filename
            }
            self.on_new_violation(violation_data)

    def close(self):
        pass