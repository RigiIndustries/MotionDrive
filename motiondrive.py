import cv2
import mediapipe as mp
import math
import serial
import time

# === SERIAL SETUP ===
arduino = serial.Serial('COM6', 9600, timeout=1)
time.sleep(2)
last_command = None

# === MEDIAPIPE SETUP ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# === CAMERA SETUP ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

# === WINDOW SETUP ===
screen_w = 1920
screen_h = 1080
cv2.namedWindow("MotionDrive", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MotionDrive", screen_w, screen_h)

# === STATES ===
STATE_TITLE = 0
STATE_TUTORIAL = 1
STATE_DRIVE = 2
state = STATE_TITLE
tutorial_step = 0
tutorial_done = False
tutorial_last_switch = 0
just_switched_to_tutorial = False
gesture_hold_start = None

ROLL_THRESHOLDS = {
    'forward':  {'left': 0.25, 'right': 1.4},
    'backward': {'left': 0.6,  'right': 1.8},
    'neutral':  {'left': 0.3,  'right': 1.5}
}

TUTORIAL_STEPS = [
    ("Make a fist", "This means STOP."),
    ("Tilt your hand left", "This steers the vehicle left."),
    ("Tilt your hand right", "This steers right."),
    ("Open your palm", "This makes the vehicle go forward."),
    ("Point with your index finger", "This makes the vehicle reverse.")
]

tutorial_images = [
    cv2.flip(cv2.imread("Tutorial_Images/tut1.png"), 1),
    cv2.flip(cv2.imread("Tutorial_Images/tut4.png"), 1),
    cv2.flip(cv2.imread("Tutorial_Images/tut2.png"), 1),
    cv2.flip(cv2.imread("Tutorial_Images/tut6.png"), 1),
    cv2.flip(cv2.imread("Tutorial_Images/tut5.png"), 1)
]

def fingers_up(landmarks):
    return [
        landmarks[8].y < landmarks[6].y,
        landmarks[12].y < landmarks[10].y,
        landmarks[16].y < landmarks[14].y,
        landmarks[20].y < landmarks[18].y
    ]

def get_hand_roll(landmarks):
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    dx = pinky_mcp.x - index_mcp.x
    dy = pinky_mcp.y - index_mcp.y
    return math.atan2(dy, dx)

def draw_centered_text(frame, text, y, scale=1.5, color=(255, 255, 255), thickness=3, outline=True):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (frame.shape[1] - text_size[0]) // 2
    if outline:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return y + text_size[1] + 10

def draw_info(frame, accel_cmd, steer_cmd, roll_angle, mode):
    cv2.putText(frame, f"{mode} | A: {accel_cmd or '-'} | S: {steer_cmd or '-'} | R: {roll_angle:.2f}",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    text = "ESC: Back to title"
    pos = (30, frame.shape[0] - 30)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)  # outline
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)  # fill


def resize_with_letterbox(frame, screen_w, screen_h):
    h, w = frame.shape[:2]
    frame_aspect = w / h
    screen_aspect = screen_w / screen_h

    if frame_aspect > screen_aspect:
        new_w = screen_w
        new_h = int(screen_w / frame_aspect)
    else:
        new_h = screen_h
        new_w = int(screen_h * frame_aspect)

    resized = cv2.resize(frame, (new_w, new_h))
    pad_top = (screen_h - new_h) // 2
    pad_bottom = screen_h - new_h - pad_top
    pad_left = (screen_w - new_w) // 2
    pad_right = screen_w - new_w - pad_left

    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def draw_hand_skeleton(frame, landmarks, connections, color=(255, 255, 255)):
    for lm in landmarks:
        h, w = frame.shape[:2]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, color, -1)
    for conn in connections:
        p1 = landmarks[conn[0]]
        p2 = landmarks[conn[1]]
        x1, y1 = int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0])
        x2, y2 = int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

def send_combined_command(accel_cmd, steer_cmd):
    global last_command
    if accel_cmd == 'f' and steer_cmd == 'r': combo = 'a'
    elif accel_cmd == 'f' and steer_cmd == 'l': combo = 'z'
    elif accel_cmd == 'b' and steer_cmd == 'r': combo = 'y'
    elif accel_cmd == 'b' and steer_cmd == 'l': combo = 'w'
    elif accel_cmd: combo = accel_cmd
    elif steer_cmd: combo = steer_cmd
    else: combo = 'x'
    if combo != last_command:
        arduino.write(combo.encode())
        last_command = combo

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    hand_detected = result.multi_hand_landmarks is not None

    accel_cmd = ''
    steer_cmd = ''
    gesture_type = 'neutral'
    roll_angle = 0
    fingers = [False, False, False, False]

    if hand_detected:
        landmarks = result.multi_hand_landmarks[0].landmark
        fingers = fingers_up(landmarks)
        roll_angle = get_hand_roll(landmarks)
        if all(fingers):
            accel_cmd = 'f'
            gesture_type = 'forward'
        elif fingers[0] and not any(fingers[1:]):
            accel_cmd = 'b'
            gesture_type = 'backward'
        thresholds = ROLL_THRESHOLDS[gesture_type]
        if roll_angle < thresholds['left']: steer_cmd = 'l'
        elif roll_angle > thresholds['right']: steer_cmd = 'r'

    if state == STATE_TITLE:
        overlay = frame.copy()
        overlay[:] = (0, 0, 0)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        draw_centered_text(frame, "MotionDrive", frame.shape[0] // 2 - 60, 5.0)
        draw_centered_text(frame, "Press SPACE to start", frame.shape[0] // 2 + 30, 1.5, (200, 200, 200))
        draw_centered_text(frame, "Press S to skip tutorial", frame.shape[0] // 2 + 90, 1.2, (180, 180, 180))

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            state = STATE_TUTORIAL
            gesture_hold_start = None
            tutorial_step = 0
            tutorial_done = False
            tutorial_last_switch = 0
            just_switched_to_tutorial = True
        elif key == ord('s'):
            state = STATE_DRIVE
        elif key == 27:
            break

    elif state == STATE_TUTORIAL:
        if just_switched_to_tutorial:
            tutorial_last_switch = time.time()
            just_switched_to_tutorial = False
            screen = resize_with_letterbox(frame, screen_w, screen_h)
            cv2.imshow("MotionDrive", screen)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            state = STATE_TITLE
            continue

        if tutorial_step < len(tutorial_images):
            img = tutorial_images[tutorial_step]
            if img is not None:
                thumb_h = 500
                scale = thumb_h / img.shape[0]
                thumb_w = int(img.shape[1] * scale)
                resized = cv2.resize(img, (thumb_w, thumb_h))

                margin = 30  # space from the window edge
                x_offset = margin
                y_offset = frame.shape[0] - thumb_h - margin - 50

                frame[y_offset:y_offset + thumb_h, x_offset:x_offset + thumb_w] = resized

        step_done = False
        if tutorial_step == 0 and hand_detected and not any(fingers):
            neutral_left = ROLL_THRESHOLDS['neutral']['left']
            neutral_right = ROLL_THRESHOLDS['neutral']['right']
            neutral_center = (neutral_left + neutral_right) / 2
            tolerance = 0.6
            if neutral_center - tolerance <= roll_angle <= neutral_center + tolerance:
                step_done = True
        elif tutorial_step == 1 and steer_cmd == 'l':
            step_done = True
        elif tutorial_step == 2 and steer_cmd == 'r':
            step_done = True
        elif tutorial_step == 3 and accel_cmd == 'f':
            step_done = True
        elif tutorial_step == 4 and accel_cmd == 'b':
            step_done = True

        if hand_detected:
            hand_color = (255, 255, 255)  # default white
            if step_done:
                hand_color = (0, 255, 0)  # green if correct gesture detected
            draw_hand_skeleton(frame, result.multi_hand_landmarks[0].landmark, mp_hands.HAND_CONNECTIONS,
                               color=hand_color)

        if step_done:
            if gesture_hold_start is None:
                gesture_hold_start = time.time()
            elif time.time() - gesture_hold_start > 2:
                tutorial_step += 1
                tutorial_last_switch = time.time()
                gesture_hold_start = None
        else:
            gesture_hold_start = None

        if tutorial_step < len(TUTORIAL_STEPS):
            step_title, step_desc = TUTORIAL_STEPS[tutorial_step]
            draw_centered_text(frame, step_title, frame.shape[0] // 2 - 60, 2.0)
            draw_centered_text(frame, step_desc, frame.shape[0] // 2, 1.2, (200, 200, 200))
            text = f"Step {tutorial_step+1} of {len(TUTORIAL_STEPS)}"
            cv2.putText(frame, text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)      # black outline
            cv2.putText(frame, text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)  # light grey fill

        else:
            if not tutorial_done:
                tutorial_done = True
                tutorial_last_switch = time.time()
            elif time.time() - tutorial_last_switch > 2:
                state = STATE_DRIVE
            else:
                overlay = frame.copy()
                overlay[:] = (0, 0, 0)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                draw_centered_text(frame, "Tutorial Complete.  Get Ready...", frame.shape[0] // 2, 2.0)

        draw_info(frame, accel_cmd, steer_cmd, roll_angle, "Tutorial")

    elif state == STATE_DRIVE:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            state = STATE_TITLE
            send_combined_command('', '')
            continue
        if hand_detected:
            hand_color = (255, 255, 255)
            draw_hand_skeleton(frame, result.multi_hand_landmarks[0].landmark, mp_hands.HAND_CONNECTIONS, color=hand_color)

        send_combined_command(accel_cmd, steer_cmd)
        draw_info(frame, accel_cmd, steer_cmd, roll_angle, "Drive")

    screen = resize_with_letterbox(frame, screen_w, screen_h)
    cv2.imshow("MotionDrive", screen)

cap.release()
cv2.destroyAllWindows()
