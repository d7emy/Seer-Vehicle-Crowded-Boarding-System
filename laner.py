import cv2
import numpy as np

video_path = 'Rm cut 2.mp4'
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()

if not success:
    print("Error: Could not read video.")
    exit()

all_lanes = []
current_lane_points = []
redo_stack = [] # Stores points for Redo

def click_event(event, x, y, flags, params):
    global current_lane_points, redo_stack
    if event == cv2.EVENT_LBUTTONDOWN:
        current_lane_points.append([x, y])
        redo_stack.clear() # New action clears redo history
        draw_refresh()

def draw_refresh():
    global img_display
    img_display = frame.copy()
    
    # 1. Draw completed lanes (Green)
    for i, lane in enumerate(all_lanes):
        pts = np.array(lane, np.int32)
        cv2.polylines(img_display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img_display, f"Lane {i+1}", tuple(lane[0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. Draw current lane (Red)
    if len(current_lane_points) > 0:
        for pt in current_lane_points:
            cv2.circle(img_display, tuple(pt), 4, (0, 0, 255), -1)
        
        pts = np.array(current_lane_points, np.int32)
        if len(current_lane_points) >= 2:
            cv2.polylines(img_display, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.imshow("Multi-Lane Builder", img_display)

img_display = frame.copy()
cv2.namedWindow("Multi-Lane Builder")
cv2.setMouseCallback("Multi-Lane Builder", click_event)

print("--- CONTROLS ---")
print("LEFT CLICK: Add point")
print("'u': UNDO last point | 'i': REDO last point")
print("'n': SAVE lane       | 'r': RESET all")
print("'q': FINISH & PRINT")

while True:
    cv2.imshow("Multi-Lane Builder", img_display)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('u'): # Undo
        if current_lane_points:
            redo_stack.append(current_lane_points.pop())
            draw_refresh()
            
    elif key == ord('i'): # Redo (using 'i' as it's next to 'u')
        if redo_stack:
            current_lane_points.append(redo_stack.pop())
            draw_refresh()

    elif key == ord('n'): # New Lane
        if len(current_lane_points) >= 3:
            all_lanes.append(current_lane_points)
            current_lane_points = []
            redo_stack.clear()
            print(f"Lane {len(all_lanes)} saved.")
            draw_refresh()

    elif key == ord('r'): # Reset
        all_lanes = []
        current_lane_points = []
        redo_stack.clear()
        draw_refresh()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final output
print("\nall_lanes = [")
for lane in all_lanes:
    print(f"    np.array({lane}, np.int32),")
print("]")