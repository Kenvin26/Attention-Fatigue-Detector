# Attention Fatigue Detector

## Overview
The Attention Fatigue Detector is a real-time web application that uses your webcam to monitor signs of attention fatigue, such as blinks and yawns. It provides live metrics, a fatigue score, and a trend chart to help you understand your alertness over time. The app is built with Streamlit and leverages computer vision and machine learning libraries for robust facial analysis.

## Features
- **Live Webcam Feed:** Real-time video processing in your browser.
- **Blink & Yawn Detection:** Uses facial landmarks to detect blinks and yawns accurately.
- **Fatigue Score:** Calculates a fatigue score based on your blink and yawn frequency.
- **Trend Chart:** Visualizes your fatigue score over time.
- **Adjustable Detection Settings:** Tune blink/yawn thresholds and durations for best results.
- **Auto-Logging:** Fatigue data is saved automatically every 30 seconds.
- **Visual Feedback:** See detection confidence and real-time metrics overlaid on the video.

## Tech Stack & Why Used
- **Python:** Main programming language for rapid prototyping and scientific computing.
- **Streamlit:** For building interactive, real-time web apps with minimal code.
- **streamlit-webrtc:** Enables real-time webcam streaming and processing in Streamlit.
- **OpenCV:** For image processing and drawing overlays on video frames.
- **MediaPipe:** For robust, real-time facial landmark detection (eyes, mouth, etc.).
- **NumPy:** Efficient numerical operations for EAR/MAR calculations.
- **Pandas:** For logging, saving, and visualizing fatigue data.

## Project Structure
```
Attention-Fatigue-Detector/
  attention-fatigue-detector/
    ├── app.py                # (if present) Entry point for other app logic
    ├── dashboard.py          # Main Streamlit app (run this file)
    ├── requirements.txt      # All Python dependencies
    ├── README.md             # This file
    ├── data/
    │   └── fatigue_logs.csv  # Fatigue logs (auto-generated)
    └── ...                   # Other source files, utils, models, etc.
  venv/                       # (DO NOT UPLOAD) Your Python virtual environment
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Attention-Fatigue-Detector/attention-fatigue-detector
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   streamlit run dashboard.py
   ```
5. **Open your browser:**
   Go to the URL shown in the terminal (usually http://localhost:8501)

## Usage
- Allow camera access when prompted.
- Watch your live metrics and fatigue score update in real time.
- Adjust detection settings in the sidebar for best results.
- Fatigue data is logged automatically every 30 seconds.
- Use the trend chart to monitor your alertness over time.
```

## Credits
- Built with ❤️ using OpenCV, MediaPipe, Streamlit, and streamlit-webrtc.

## License
This project is for educational and personal use. Please check the licenses of the included libraries for commercial use.
