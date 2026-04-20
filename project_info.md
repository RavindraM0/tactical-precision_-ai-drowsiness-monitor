# Project Portfolio: Tactical Precision Monitoring

## 📋 Project Description (Resume-Ready)
**Title**: Real-time AI Drowsiness & Attention Monitoring System
**Stack**: Computer Vision, MediaPipe, React/TypeScript, Python, OpenCV.

*   Designed and implemented a real-time biometric monitoring system for industrial safety, achieving sub-15ms inference latency using MediaPipe FaceMesh.
*   Engineered a multi-stage detection pipeline calculating Eye Aspect Ratio (EAR) and Head Pose euler approximations to identify operator fatigue and distraction.
*   Developed a "Tactical Precision" UI following Industrial Brutalism principles, utilizing tonal layering and ambient lift for high-density information display without visual fatigue.
*   Implemented multi-modal alert systems with non-linear feedback loops to ensure operator engagement in safety-critical environments.

---

## 🎙️ Interview Preparation (Vivas)

### 1. What is EAR and why use it over simple eye detection?
**Answer**: EAR (Eye Aspect Ratio) provides a numerical value for eye openness that is invariant to face size and distance from the camera. Unlike simple detection, EAR allows us to differentiate between a blink and a sustained eye closure (drowsiness) by tracking the ratio over a window of frames.

### 2. How did you handle lighting and camera noise?
**Answer**: I used MediaPipe's FaceMesh which is robust against moderate lighting changes. For software-side smoothing, I implemented a frame-buffer (Consecutive Frames check) to prevent noise from triggering false-positive alerts on simple blinks or momentary occlusions.

### 3. What are the limitations of your current head pose approach?
**Answer**: The current implementation uses a geometric approximation based on landmark asymmetry. While excellent for performance, a more robust method in a production "v2" would involve a 3D generic model alignment (PnP algorithm) for actual 3D rotation vectors.

### 4. Why did you choose Industrial Brutalism for the UI?
**Answer**: In safety-critical systems, visual clutter is a hazard. Brutalism emphasizes raw functionality—heavy contrast and clear hierarchy. By removing dividers (No-Line philosophy) and using background shifts, we create a UI that stays in the background until an emergency occurs, reducing operator "alert fatigue."

---

## 🚀 Future Improvements
1.  **Person-Specific Calibration**: Auto-calculate "base EAR" for each user during a 10-second calibration phase.
2.  **Yawning Detection**: Track Lip Aspect Ratio (LAR) to detect fatigue *before* eyes start closing.
3.  **Local Edge Deployment**: Deploy on NVIDIA Jetson / Raspberry Pi for vehicle-mounted implementation.
