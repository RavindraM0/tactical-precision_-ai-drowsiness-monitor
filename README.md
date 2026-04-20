# AI Driven Monitoring System: Tactical Precision I.F.

This project is a high-performance, real-time computer vision system designed for safety-critical environments. It monitors user biometrics via a webcam to detect drowsiness (via Eye Aspect Ratio) and distracted behavior (via Head Pose Estimation).

## 🚀 Key Features
- **Industrial Brutalist HUD**: Zero-border, depth-layered interface for maximum focus.
- **Biometric Telemetry**: Live tracking of EAR (Eye Aspect Ratio), Yaw, and Pitch.
- **Multi-Modal Alerts**: Sound-frequency shifts and high-contrast visual warnings.
- **Edge Processing**: Uses MediaPipe FaceMesh for low-latency browser-based inference.

## 🛠️ Tech Stack
- **Frontend**: React 19, Tailwind CSS 4, Motion (Framer Motion).
- **Vision**: MediaPipe Tasks Vision (Face Landmarker).
- **Desktop (Portable)**: Python 3.10+, OpenCV, MediaPipe.

## 📖 Scientific Implementation

### 1. Drowsiness Detection (EAR)
The system calculates the **Eye Aspect Ratio (EAR)** using 6 facial landmarks for each eye.
Formula: `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
When EAR falls below `0.22` for `15` consecutive frames, the system triggers a **DROWSY** alert.

### 2. Attention Monitoring (Head Pose)
By analyzing the asymmetry between the nose tip and the inner corners of the eyes, we approximate head rotation.
- **Yaw**: Horizontal rotation (Looking left/right).
- **Pitch**: Vertical rotation (Looking up/down).

## 🏃 Running the Application

### Web (Current Interface)
1. Navigate to the preview.
2. Click **Initialize Watch** (Required for Audio/Camera permissions).
3. The system will load the models and start monitoring.

### Desktop (Local Python)
If you wish to run this as a native desktop app:
1. Ensure Python is installed.
2. `pip install opencv-python mediapipe numpy`
3. Run `python app.py`

## ⚠️ Limitations
- **Lighting**: Poor lighting can reduce landmark accuracy.
- **Occlusion**: Glasses (especially thick frames) may interfere with EAR calculation.
- **Static Thresholds**: Fixed thresholds may need adjustment per user (Calibration routine recommended).
