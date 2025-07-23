import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import pandas as pd
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(layout="wide", page_title="Attention Fatigue Detector")
st.markdown("""
    <style>
    .big-font {font-size:30px !important; font-weight: bold;}
    .fatigue-metric {font-size:22px !important;}
    .stButton>button {height: 3em; width: 100%; font-size: 18px;}
    .stMetric {font-size: 22px;}
    </style>
""", unsafe_allow_html=True)

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14]

def calculate_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth):
    A = np.linalg.norm(mouth[0] - mouth[1])
    C = np.linalg.norm(mouth[2] - mouth[3])
    return A / C

# --- UI and Fatigue Logic Improvements ---

def calculate_fatigue_score(blinks, yawns):
    # Example: fatigue score out of 100
    return min(100, blinks * 2 + yawns * 5)

class VideoProcessor(VideoProcessorBase):
    def __init__(self, mar_thresh=0.7, min_frames=12, cooldown=30, ear_thresh=0.25):
        self.blink_count = 0
        self.yawn_count = 0
        self.blink_thresh = ear_thresh
        self.consecutive_frames = 2
        self.blink_counter = 0
        self.yawn_cooldown = 0
        self.mar_history = []
        self.mar_window = 15  # Longer window for more smoothing
        self.ear_history = []
        self.ear_window = 5  # Smoother blink detection
        self.yawn_mar_thresh = mar_thresh
        self.yawn_min_frames = min_frames
        self.yawn_cooldown_setting = cooldown
        self.yawn_active = False
        self.yawn_frame_count = 0
        self.mar_value = 0.0
        self.ear_value = 0.0
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.last_blink_time = time.time()
        self.blink_cooldown = 1.0  # seconds
        self.last_yawn_time = time.time()
        self.yawn_cooldown_secs = 10  # seconds
        self.face_detected = True
        self.confidence = 1.0
        self.mouth_open_start = None  # For time-based yawn detection

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if result.multi_face_landmarks:
            self.face_detected = True
            for face in result.multi_face_landmarks:
                landmarks = face.landmark
                h, w, _ = img.shape
                left_eye = np.array([[landmarks[p].x * w, landmarks[p].y * h] for p in LEFT_EYE])
                right_eye = np.array([[landmarks[p].x * w, landmarks[p].y * h] for p in RIGHT_EYE])
                mouth = np.array([[landmarks[p].x * w, landmarks[p].y * h] for p in MOUTH])
                left_ear = calculate_EAR(left_eye)
                right_ear = calculate_EAR(right_eye)
                mar = calculate_MAR(mouth)
                avg_ear = (left_ear + right_ear) / 2.0
                self.ear_value = avg_ear
                self.mar_value = mar
                # --- Smoothed EAR for blink detection ---
                self.ear_history.append(avg_ear)
                if len(self.ear_history) > self.ear_window:
                    self.ear_history.pop(0)
                ear_avg = np.mean(self.ear_history)
                # --- Blink detection with cooldown ---
                if ear_avg < self.blink_thresh:
                    self.blink_counter += 1
                else:
                    if self.blink_counter >= self.consecutive_frames:
                        if time.time() - self.last_blink_time > self.blink_cooldown:
                            self.blink_count += 1
                            self.last_blink_time = time.time()
                    self.blink_counter = 0
                # --- Smoothed MAR for yawn detection ---
                self.mar_history.append(mar)
                if len(self.mar_history) > self.mar_window:
                    self.mar_history.pop(0)
                mar_avg = np.mean(self.mar_history)
                # --- Time-based yawn detection: mouth open for >2s ---
                now = time.time()
                if mar_avg > self.yawn_mar_thresh:
                    if self.mouth_open_start is None:
                        self.mouth_open_start = now
                    elif (now - self.mouth_open_start > 2 and
                          (not self.yawn_active) and
                          now - self.last_yawn_time > self.yawn_cooldown_secs):
                        self.yawn_count += 1
                        self.last_yawn_time = now
                        self.yawn_active = True
                else:
                    self.mouth_open_start = None
                    self.yawn_active = False
                    self.yawn_frame_count = 0
                if self.yawn_cooldown > 0:
                    self.yawn_cooldown -= 1
                # Confidence overlay (simple: always 1.0 if face detected)
                self.confidence = 1.0
        else:
            self.face_detected = False
            self.confidence = 0.0
            cv2.putText(img, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # UI overlay
        cv2.rectangle(img, (0, 0), (350, 170), (245, 245, 245), -1)
        cv2.putText(img, f"Blinks: {self.blink_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 102, 204), 2)
        cv2.putText(img, f"Yawns: {self.yawn_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (204, 102, 0), 2)
        cv2.putText(img, f"MAR: {self.mar_value:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"EAR: {self.ear_value:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        mar_bar_x = 200
        mar_bar_y = 120
        mar_bar_length = int(100 * min(self.mar_value, 1.5))
        cv2.rectangle(img, (mar_bar_x, mar_bar_y-10), (mar_bar_x+mar_bar_length, mar_bar_y+10), (0, 180, 0), -1)
        thresh_pos = int(mar_bar_x + 100 * self.yawn_mar_thresh)
        cv2.line(img, (thresh_pos, mar_bar_y-15), (thresh_pos, mar_bar_y+15), (0, 0, 255), 2)
        # Confidence overlay
        conf_color = (0, 200, 0) if self.confidence > 0.5 else (0, 0, 255)
        cv2.putText(img, f"Confidence: {self.confidence:.2f}", (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Initialize session state for yawn detection settings
    if "mar_thresh" not in st.session_state:
        st.session_state["mar_thresh"] = 0.7
    if "min_frames" not in st.session_state:
        st.session_state["min_frames"] = 12
    if "cooldown" not in st.session_state:
        st.session_state["cooldown"] = 30
    st.title("🧠 Attention Fatigue Detector")
    st.markdown("<hr style='margin:0 0 20px 0;border:1px solid #eee;'>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1], gap="large")
    with col2:
        st.markdown('<div class="big-font">Yawn Detection Settings</div>', unsafe_allow_html=True)
        mar_thresh = st.slider("Yawn MAR Threshold", min_value=0.5, max_value=1.2, value=0.7, step=0.01, key="mar_thresh")
        min_frames = st.slider("Min Frames for Yawn", min_value=3, max_value=30, value=12, step=1, key="min_frames")
        cooldown = st.slider("Yawn Cooldown (frames)", min_value=5, max_value=60, value=30, step=1, key="cooldown")
        ear_thresh = st.slider("Blink EAR Threshold", min_value=0.1, max_value=0.35, value=0.25, step=0.01, key="ear_thresh")
    with col1:
        st.markdown('<div class="big-font">Live Webcam Feed</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="fatigue-detect",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: VideoProcessor(
                mar_thresh=mar_thresh,
                min_frames=min_frames,
                cooldown=cooldown,
                ear_thresh=ear_thresh
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    # --- Auto-save logs every 30 seconds ---
    if 'last_log_time' not in st.session_state:
        st.session_state['last_log_time'] = time.time()
    if ctx and ctx.video_processor and time.time() - st.session_state['last_log_time'] > 30:
        blinks = ctx.video_processor.blink_count
        yawns = ctx.video_processor.yawn_count
        fatigue_score = calculate_fatigue_score(blinks, yawns)
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, 'fatigue_logs.csv')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = pd.DataFrame([{"timestamp": timestamp, "blinks": blinks, "yawns": yawns, "score": fatigue_score}])
        if not os.path.exists(csv_path):
            log.to_csv(csv_path, index=False)
        else:
            log.to_csv(csv_path, mode="a", header=False, index=False)
        st.session_state['last_log_time'] = time.time()
    with col2:
        st.markdown('<div class="big-font">Fatigue Metrics</div>', unsafe_allow_html=True)
        if ctx.video_processor:
            blinks = ctx.video_processor.blink_count
            yawns = ctx.video_processor.yawn_count
            fatigue_score = calculate_fatigue_score(blinks, yawns)
            st.metric(label="Blinks", value=blinks, delta=None)
            st.metric(label="Yawns", value=yawns, delta=None)
            st.metric(label="Fatigue Score (Live)", value=f"{fatigue_score}/100", delta=None)
            # Real-time trend chart (in-memory, not just on button click)
            # Load or create fatigue trend data in session state
            if 'fatigue_trend' not in st.session_state:
                st.session_state['fatigue_trend'] = []
            # Append new score if changed
            if (not st.session_state['fatigue_trend']) or (st.session_state['fatigue_trend'][-1][1] != fatigue_score):
                st.session_state['fatigue_trend'].append((datetime.now().strftime("%H:%M:%S"), fatigue_score))
            # Show trend chart
            trend_df = pd.DataFrame(st.session_state['fatigue_trend'], columns=["time", "score"])
            trend_df = trend_df.tail(30)  # Show last 30 points
            st.line_chart(trend_df.set_index("time")['score'], height=200)
            if st.button("Check Fatigue Score", key="check_fatigue_btn"):
                st.success(f"Fatigue check complete! Blinks: {blinks}, Yawns: {yawns}, Score: {fatigue_score}")
                # Save to logs
                data_dir = os.path.join(os.path.dirname(__file__), 'data')
                os.makedirs(data_dir, exist_ok=True)
                csv_path = os.path.join(data_dir, 'fatigue_logs.csv')
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log = pd.DataFrame([{"timestamp": timestamp, "blinks": blinks, "yawns": yawns, "score": fatigue_score}])
                if not os.path.exists(csv_path):
                    log.to_csv(csv_path, index=False)
                else:
                    log.to_csv(csv_path, mode="a", header=False, index=False)
        st.markdown("---")
        # Historical fatigue trend chart from CSV
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        csv_path = os.path.join(data_dir, 'fatigue_logs.csv')
        if os.path.exists(csv_path):
            # Try to read with header, fallback to no header if columns missing
            try:
                df = pd.read_csv(csv_path)
                if 'blinks' not in df.columns or 'yawns' not in df.columns:
                    raise ValueError
            except Exception:
                # Assume no header, assign names
                df = pd.read_csv(csv_path, header=None, names=["timestamp", "blinks", "yawns", "score"])
            if 'score' not in df.columns:
                df['score'] = df['blinks'] * 2 + df['yawns'] * 5
            if not df.empty and 'timestamp' in df.columns and 'score' in df.columns:
                # Robust timestamp parsing: strip spaces, infer format, coerce errors
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.strip(), errors='coerce', infer_datetime_format=True)
                df["time"] = df["timestamp"].dt.strftime("%H:%M")
                st.line_chart(df.set_index("time")["score"], height=200)
            else:
                st.info("No trend data yet. Run the fatigue tracker first.")
        else:
            st.info("No trend data yet. Run the fatigue tracker first.")
    st.caption("<div style='text-align:center'>Made with ❤️ using OpenCV, MediaPipe, and Streamlit + streamlit-webrtc</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 