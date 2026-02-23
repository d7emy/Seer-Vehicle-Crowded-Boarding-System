import cv2


from storage_manager import StorageManager
from vision_tracker import VisionTracker
from violation_engine import ViolationEngine

class MainSystem:
    def __init__(self, video_path):
        self.video_path = video_path
        #fps must be modified when input changed
        self.fps = 30
        
        # تهيئة الكلاسات وربط المحرك بمدير التخزين
        self.storage = StorageManager()
        self.tracker = VisionTracker(fps=self.fps)
        self.engine = ViolationEngine(self.storage)
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
            
            # 1. التتبع وحساب السرعة
            tracked_objects = self.tracker.track_and_get_speeds(frame, current_time)
            
            # 2. تطبيق القوانين، عرض العداد التنازلي، وحفظ المخالفات
            processed_frame = self.engine.process_and_draw(frame, tracked_objects, current_time)
            
            # 3. العرض النهائي
            cv2.imshow("Seer-Vehicle Crowded Boarding System", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.storage.close()

if __name__ == "__main__":
    # تأكد أن ملف الفيديو موجود في نفس المجلد
    app = MainSystem('Rm cut 2.mp4')
    app.run()