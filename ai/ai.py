import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas and color settings
canvas = None
brush_size = 10
hand_colors = {}
drawing_enabled = {}
prev_positions = {}

color_options = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]  # Red, Green, Blue, Eraser
clear_button_region = (340, 20, 400, 80)

def detect_finger_tips(results, width, height):
    finger_tips = {}
    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):        
            finger_tips[hand_index] = {
                "index": (
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width),
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                ),
                "middle": (
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width),
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
                )
            }
    return finger_tips

def check_pause_condition(finger_tips):
    ix, iy = finger_tips["index"]
    mx, my = finger_tips["middle"]
    distance = np.sqrt((ix - mx)**2 + (iy - my)**2)
    return distance < 50

def draw_skeleton_and_points(frame, results, width, height):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for landmark_id in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP]:
                cx = int(hand_landmarks.landmark[landmark_id].x * width)
                cy = int(hand_landmarks.landmark[landmark_id].y * height)
                cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if canvas is None:
        canvas = np.zeros_like(frame)

    results = hands.process(rgb_frame)

    # Draw color selection boxes and clear button
    for i, color in enumerate(color_options):
        x1, y1 = 20 + i * 80, 20
        x2, y2 = x1 + 60, y1 + 60
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (340, 20), (400, 80), (0, 0, 0), -1)
    cv2.putText(frame, "Clear", (345, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    draw_skeleton_and_points(frame, results, width, height)

    all_finger_tips = detect_finger_tips(results, width, height)

    for hand_index, finger_tips in all_finger_tips.items():
        if hand_index not in drawing_enabled:
            drawing_enabled[hand_index] = True
            prev_positions[hand_index] = None
            hand_colors[hand_index] = (0, 255, 0)

        if check_pause_condition(finger_tips):
            drawing_enabled[hand_index] = False
            prev_positions[hand_index] = None
            ix, iy = finger_tips["index"]
            cv2.circle(frame, (ix, iy), 15, (0, 0, 255), -1)
        else:
            drawing_enabled[hand_index] = True

        ix, iy = finger_tips["index"]
        for i, color in enumerate(color_options):
            x1, y1 = 20 + i * 80, 20
            x2, y2 = x1 + 60, y1 + 60
            if x1 < ix < x2 and y1 < iy < y2:
                hand_colors[hand_index] = color
        if clear_button_region[0] < ix < clear_button_region[2] and clear_button_region[1] < iy < clear_button_region[3]:
            canvas = np.zeros_like(frame)

        if drawing_enabled[hand_index]:
            x, y = finger_tips["index"]
            color = hand_colors[hand_index]

            if prev_positions[hand_index]:
                cv2.line(canvas, prev_positions[hand_index], (x, y), color, brush_size)
            prev_positions[hand_index] = (x, y)

    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Create a named window with adjustable properties BEFORE using it
    cv2.namedWindow("MaiBok", cv2.WINDOW_NORMAL)

    # Set a specific size for the window (optional)
    cv2.resizeWindow("MaiBok", 1280, 720)  # Width x Height

    # Show the combined frame
    cv2.imshow("MaiBok", combined)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
