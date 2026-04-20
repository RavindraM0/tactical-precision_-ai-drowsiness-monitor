import cv2
import mediapipe as mp
import numpy as np
import time
import math
import threading
import winsound

# --- Configuration ---
EAR_THRESHOLD = 0.28
YAW_THRESHOLD = 20
PITCH_THRESHOLD = 15
CONSECUTIVE_FRAMES = 8

# MediaPipe Face Mesh Initialization
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def compute_ear(landmarks, w, h):
    """Compute Eye Aspect Ratio (EAR) in Pixel Space"""
    def distance(p1, p2):
        return math.sqrt(((p1.x - p2.x) * w)**2 + ((p1.y - p2.y) * h)**2)

    # Right Eye indices
    r1 = distance(landmarks[33], landmarks[133])
    r2 = distance(landmarks[160], landmarks[144])
    r3 = distance(landmarks[158], landmarks[153])
    r_ear = (r2 + r3) / (2.0 * r1)

    # Left Eye indices
    l1 = distance(landmarks[362], landmarks[263])
    l2 = distance(landmarks[385], landmarks[380])
    l3 = distance(landmarks[387], landmarks[373])
    l_ear = (l2 + l3) / (2.0 * l1)

    return (r_ear + l_ear) / 2.0

def estimate_head_pose(landmarks, w, h):
    """Estimate head direction in Pixel Space"""
    nose = landmarks[1]
    l_eye = landmarks[133]
    r_eye = landmarks[362]
    
    # Simple horizontal ratio for yaw
    d_l = abs((nose.x - l_eye.x) * w)
    d_r = abs((nose.x - r_eye.x) * w)
    yaw = ((d_l - d_r) / (d_l + d_r + 1e-6)) * 100
    
    # Simple vertical offset for pitch
    mid_eye_y = ((l_eye.y + r_eye.y) / 2) * h
    pitch = (mid_eye_y - (nose.y * h)) * 0.5
    
    return yaw, pitch

def play_alert():
    """Play alert sound asynchronously"""
    winsound.Beep(1000, 200)

def main():
    cap = cv2.VideoCapture(0)
    drowsy_counter = 0
    last_alert_time = 0
    eyes_already_closed = False
    start_time = time.time()
    frames = 0
    fps = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Warning: [Camera] Failed to read frame. Retrying...")
            continue

        # Flip for mirror effect and convert to RGB
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        status_text = "AWAKE"
        status_color = (0, 255, 65) # Neon Green
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = image.shape
            
            # 1. EAR Logic
            ear = compute_ear(landmarks, w, h)
            if ear < EAR_THRESHOLD:
                drowsy_counter += 1
                # Blink detected (first frame of closure)
                if not eyes_already_closed:
                    threading.Thread(target=play_alert, daemon=True).start()
                    eyes_already_closed = True
            else:
                drowsy_counter = 0
                eyes_already_closed = False
            
            # 2. Attention Logic
            yaw, pitch = estimate_head_pose(landmarks, w, h)
            distracted = abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD

            # 3. Decision Logic
            if drowsy_counter > CONSECUTIVE_FRAMES:
                status_text = "CRITICAL: DROWSY!"
                status_color = (0, 0, 196) # Red (BGR)
                # Alert sound every 0.5 seconds
                if time.time() - last_alert_time > 0.5:
                    threading.Thread(target=play_alert, daemon=True).start()
                    last_alert_time = time.time()
            elif distracted:
                status_text = "DISTRACTED!"
                status_color = (0, 191, 255) # Warning Amber (BGR)

            # --- Industrial HUD overlay ---
            h, w, _ = image.shape
            # Background dark overlays for "No-Line" feel
            cv2.rectangle(image, (0, 0), (250, 180), (19, 19, 19), -1)
            cv2.putText(image, f"EAR: {ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"YAW: {yaw:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"PITCH: {pitch:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"FPS: {fps}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 65), 1)

            # Central Big Status
            cv2.putText(image, status_text, (w//2 - 150, h - 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, status_color, 2)

        # FPS Calculation
        frames += 1
        if time.time() - start_time > 1:
            fps = frames
            frames = 0
            start_time = time.time()

        cv2.imshow('TACTICAL PRECISION SYSTEM 01', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
