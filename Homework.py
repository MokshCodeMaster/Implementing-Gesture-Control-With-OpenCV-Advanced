import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
filter_mode = 0
print("--- Keyboard Controls ---")
print("0: Normal View")
print("1: Median Blur Filter")
print("q: Quit")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    h, w, c = frame.shape
    if filter_mode == 0:
        processed_frame = frame.copy()
    elif filter_mode == 1:
        processed_frame = cv2.medianBlur(frame, 5)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                if cx > x_max: x_max = cx
                if cx < x_min: x_min = cx
                if cy > y_max: y_max = cy
                if cy < y_min: y_min = cy
            cv2.rectangle(processed_frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
    modes_text = ["Normal", "Median Blur"]
    cv2.putText(processed_frame, f"Filter: {modes_text[filter_mode]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Camera Feed', processed_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif ord('0') <= key <= ord('1'):
        filter_mode = key - ord('0')
cap.release()
hands.close()
cv2.destroyAllWindows()
