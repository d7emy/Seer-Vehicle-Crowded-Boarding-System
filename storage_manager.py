import os
import cv2
from datetime import datetime

class StorageManager:
    def __init__(self):
        # إنشاء مجلد violations وصور المخالفات بشكل ديناميكي بجانب ملف الكود
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        self.SAVE_DIR = os.path.join(self.BASE_DIR, "violations")
        self.IMG_DIR = os.path.join(self.SAVE_DIR, "images")
        
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.IMG_DIR, exist_ok=True)
        
        self.saved_snaps = set()



    def save_snapshot(self, frame, track_id, violation_type, lane_num=None, bbox=None):
        key = (int(track_id), str(violation_type))
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
        filename = os.path.join(self.IMG_DIR, f"{safe_name}_id{track_id}_{now_dt.strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, overlay)

        
        print(f"[LOG] Authority Notified: {violation_type} by Vehicle ID {track_id} at {ts_str}")
