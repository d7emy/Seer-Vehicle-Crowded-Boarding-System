import cv2
import math
import numpy as np
import uuid
from ultralytics import YOLO
from collections import defaultdict, deque

class VisionTracker:
    def __init__(self, model_path='yolov8x.pt', fps=30):
        self.model = YOLO(model_path).to('cuda')
        self.FPS = fps
        self.car_history = defaultdict(lambda: deque(maxlen=self.FPS))
        
        self.uuid_map = {} 
        
        self.SOURCE_PTS = np.array([[656, 241], [783, 235], [998, 689], [674, 694]], dtype=np.float32)
        self.TARGET_PTS = np.array([[0, 0], [7.0, 0], [7.0, 50.0], [0, 50.0]], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.SOURCE_PTS, self.TARGET_PTS)

    def track_and_get_speeds(self, frame, current_time):
        results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
        tracked_objects = []

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return tracked_objects

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, yolo_id in zip(boxes, track_ids):
            if yolo_id not in self.uuid_map:
                self.uuid_map[yolo_id] = str(uuid.uuid4())
            
            vehicle_uuid = self.uuid_map[yolo_id]

            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int(y2)

            pt = np.array([[[cx, cy]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.M)
            real_x, real_y = transformed[0][0][0], transformed[0][0][1]

            # Use the UUID for tracking history instead of the YOLO ID
            self.car_history[vehicle_uuid].append((real_x, real_y, current_time))
            speed_kmh = 0.0

            if len(self.car_history[vehicle_uuid]) > self.FPS // 2:
                old_x, old_y, old_time = self.car_history[vehicle_uuid][0]
                dist = math.hypot(real_x - old_x, real_y - old_y)
                t_diff = current_time - old_time
                if t_diff > 0:
                    speed_kmh = (dist / t_diff) * 3.6

            tracked_objects.append({
                "id": vehicle_uuid, "bbox": (x1, y1, x2, y2), "cy": cy, "speed": speed_kmh
            })

        return tracked_objects