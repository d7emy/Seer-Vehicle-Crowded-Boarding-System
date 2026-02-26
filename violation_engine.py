import cv2
import numpy as np
import json
import os

class ViolationEngine:
    def __init__(self, storage_manager):
        self.storage = storage_manager
        
        # New: Read lanes from a JSON configuration file
        self.config_file = "lanes_config.json"
        self.all_lanes = []
        self.lane_masks = []
        self.masks_initialized = False

        self.DWELL_LIMIT_SEC = 3.0
        self.STOP_WAIT_TIME_SEC = 5.0
        self.STALL_SPEED_KMH = 3
        self.LANE_CHANGE_PAIRS = {(0, 1), (1, 0)}

        # Load the lanes immediately on startup
        self.load_lanes()
        self._reset_tracking_memory()

    def _reset_tracking_memory(self):
        self.car_last_lane = {}
        self.car_stop_start = {}
        self.lane_leader = {0: {"id": None, "start_time": None}}
        self.cars_entered = set()
        self.lane_change_violators = set()
        self.stopped_violators = set()
        self.snapped_lane_change = set()
        self.snapped_stopped = set()
        self.snapped_l1_dwell = set()

    def load_lanes(self):
        """Loads lane coordinates from JSON. Called on startup and when updated via Web UI."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    lanes_data = json.load(f)
                    # Convert standard JSON lists back into OpenCV numpy arrays
                    self.all_lanes = [np.array(lane, np.int32) for lane in lanes_data]
            except Exception as e:
                print(f"Error loading lanes: {e}")
                self.all_lanes = []
        else:
            self.all_lanes = []
            
        # Force the engine to redraw the internal masks on the next frame
        self.masks_initialized = False 
        self._reset_tracking_memory()
        print(f"[ENGINE] Successfully loaded {len(self.all_lanes)} lanes from config.")

    def _init_masks(self, h, w):
        self.lane_masks = []
        for lane in self.all_lanes:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [lane], 255)
            self.lane_masks.append(mask)
        self.masks_initialized = True

    def process_and_draw(self, frame, tracked_objects, current_time):
        h, w = frame.shape[:2]
        
        if not self.masks_initialized:
            self._init_masks(h, w)

        for i, lane in enumerate(self.all_lanes):
            if len(lane) >= 2: 
                cv2.polylines(frame, [lane], isClosed=True, color=(255, 255, 0), thickness=2)
                cv2.putText(frame, f"Lane {i+1}", tuple(lane[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cars_in_lane = {i: [] for i in range(len(self.all_lanes))}

        for obj in tracked_objects:
            track_id, (x1, y1, x2, y2), speed = obj["id"], obj["bbox"], obj["speed"]
            bbox_area = max(1, (x2 - x1) * (y2 - y1))
            current_lane, max_overlap = None, 0.0

            for i, mask in enumerate(self.lane_masks):
                y1_c, y2_c = max(0, y1), min(h, y2)
                x1_c, x2_c = max(0, x1), min(w, x2)
                if y2_c <= y1_c or x2_c <= x1_c: continue
                roi = mask[y1_c:y2_c, x1_c:x2_c]
                overlap = cv2.countNonZero(roi) / float(bbox_area)
                if overlap > 0.40 and overlap > max_overlap:
                    max_overlap = overlap
                    current_lane = i
            
            obj["current_lane"] = current_lane

            prev_lane = self.car_last_lane.get(track_id)
            if prev_lane is not None and current_lane is not None and prev_lane != current_lane:
                if (prev_lane, current_lane) in self.LANE_CHANGE_PAIRS:
                    self.lane_change_violators.add(track_id)
                    if track_id not in self.snapped_lane_change:
                        self.snapped_lane_change.add(track_id)
                        self.storage.save_snapshot(frame, track_id, f"ILLEGAL LANE CHANGE L{prev_lane+1} to L{current_lane+1}", current_lane + 1, (x1, y1, x2, y2))

            if current_lane is not None:
                self.car_last_lane[track_id] = current_lane
                self.cars_entered.add(track_id)
                cars_in_lane[current_lane].append(obj)

            if current_lane is not None and current_lane != 0:
                if speed <= self.STALL_SPEED_KMH:
                    if track_id not in self.car_stop_start:
                        self.car_stop_start[track_id] = current_time
                    else:
                        stop_duration = current_time - self.car_stop_start[track_id]
                        if stop_duration > self.STOP_WAIT_TIME_SEC:
                            self.stopped_violators.add(track_id)
                            if track_id not in self.snapped_stopped:
                                self.snapped_stopped.add(track_id)
                                self.storage.save_snapshot(frame, track_id, "ILLEGAL STOPPING", current_lane + 1, (x1, y1, x2, y2))
                else:
                    if track_id in self.car_stop_start:
                        del self.car_stop_start[track_id]

        if not cars_in_lane.get(0):
            self.lane_leader[0] = {"id": None, "start_time": None}
        else:
            leader = max(cars_in_lane[0], key=lambda d: d["cy"])
            if self.lane_leader[0]["id"] != leader["id"]:
                self.lane_leader[0] = {"id": leader["id"], "start_time": current_time}

        for obj in tracked_objects:
            track_id, (x1, y1, x2, y2), speed, lane_idx = obj["id"], obj["bbox"], obj["speed"], obj["current_lane"]
            speed_str = f"{speed:.0f}km/h"
            color, label = (0, 255, 255), f"NO LANE | {speed_str}"

            if lane_idx is not None:
                lane_num = lane_idx + 1
                is_l1_leader = (lane_idx == 0) and (self.lane_leader[0]["id"] == track_id)

                if track_id in self.lane_change_violators:
                    color, label = (0, 0, 255), f"LANE VIOLATOR | {speed_str}"
                elif track_id in self.stopped_violators:
                    color, label = (0, 0, 255), f"ILLEGAL STOP L{lane_num} | {speed_str}"
                elif is_l1_leader and self.lane_leader[0]["start_time"] is not None:
                    elapsed = current_time - self.lane_leader[0]["start_time"]
                    if elapsed > self.DWELL_LIMIT_SEC:
                        color, label = (0, 0, 255), f"L1 DWELL VIOLATOR | {speed_str}"
                        if track_id not in self.snapped_l1_dwell:
                            self.snapped_l1_dwell.add(track_id)
                            self.storage.save_snapshot(frame, track_id, "EXCESSIVE WAITING TIME", 1, (x1, y1, x2, y2))
                    else:
                        color = (0, 255, 0)
                        time_left = max(0, self.DWELL_LIMIT_SEC - elapsed)
                        label = f"L1 LEADER (Timer: {time_left:.1f}s) | {speed_str}"
                else:
                    color = (0, 255, 255)
                    label = f"L{lane_num} NORMAL | {speed_str}"
            elif track_id in self.cars_entered:
                color, label = (100, 100, 100), f"EXITED | {speed_str}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            short_id = str(track_id)[:8] if track_id else ""
            cv2.putText(frame, f"{label} ID:{short_id}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame