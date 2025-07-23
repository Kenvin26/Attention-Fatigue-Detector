import cv2
import mediapipe as mp
import time
import numpy as np
from datetime import datetime
import pandas as pd
import os

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [13, 14, 78, 308, 82, 312]

def calculate_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_MAR(mouth):
    A = np.linalg.norm(mouth[0] - mouth[1])
    C = np.linalg.norm(mouth[2] - mouth[3])
    mar = A / C
    return mar

def run_webcam_fatigue(duration=60, save_logs=True):
    cap = cv2.VideoCapture(0)
    blink_count = 0
    yawn_count = 0
    start_time = time.time()
    blink_thresh = 0.25
    consecutive_frames = 2
    blink_counter = 0
    logs = []
    yawn_cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:
                landmarks = face.landmark
                h, w, _ = frame.shape
                left_eye = np.array([[landmarks[p].x * w, landmarks[p].y * h] for p in LEFT_EYE])
                right_eye = np.array([[landmarks[p].x * w, landmarks[p].y * h] for p in RIGHT_EYE])
                mouth = np.array([[landmarks[p].x * w, landmarks[p].y * h] for p in MOUTH])
                left_ear = calculate_EAR(left_eye)
                right_ear = calculate_EAR(right_eye)
                mar = calculate_MAR(mouth)
                avg_ear = (left_ear + right_ear) / 2.0
                if avg_ear < blink_thresh:
                    blink_counter += 1
                else:
                    if blink_counter >= consecutive_frames:
                        blink_count += 1
                    blink_counter = 0
                if mar > 0.7 and yawn_cooldown == 0:
                    yawn_count += 1
                    yawn_cooldown = 30
                if yawn_cooldown > 0:
                    yawn_cooldown -= 1
        elapsed = time.time() - start_time
        if elapsed > duration:
            break
    cap.release()
    cv2.destroyAllWindows()
    if save_logs:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, 'fatigue_logs.csv')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs.append({
            "timestamp": timestamp,
            "blinks": blink_count,
            "yawns": yawn_count
        })
        df = pd.DataFrame(logs)
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)
    return blink_count, yawn_count 