import cv2
import mediapipe as mp
import numpy as np
import pickle
import imageio
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Finger joint indices for each finger ──────────────────────
# Each finger: [MCP, PIP, DIP, TIP]
FINGER_TIPS   = [4, 8, 12, 16, 20]
FINGER_BASES  = [2, 5,  9, 13, 17]  # for curl calculation

class DSPGestureTrainer:
    def __init__(self):
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'hand_landmarker.task')

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.gestures         = {}
        self.is_recording     = False
        self.current_recording = []
        self.frames_for_gif   = []

        # DSP: FIR Filter (Moving Average)
        self.fir_buffer = []
        self.fir_window = 3

        self.load_gestures()

    # ── FIR smoothing on raw landmarks ────────────────────────
    def apply_fir_filter(self, landmarks):
        self.fir_buffer.append(landmarks)
        if len(self.fir_buffer) > self.fir_window:
            self.fir_buffer.pop(0)
        return np.mean(self.fir_buffer, axis=0).tolist()

    # ── Extract finger features from landmarks ────────────────
    def extract_finger_features(self, landmarks):
        lm = np.array(landmarks)  # shape (21, 3)

        features = []

        # 1. All 21 landmark positions (x, y, z) — base position
        features.extend(lm.flatten().tolist())

        # 2. Finger curl values (tip distance from base)
        #    Lower distance = more curled
        for tip, base in zip(FINGER_TIPS, FINGER_BASES):
            dist = np.linalg.norm(lm[tip] - lm[base])
            features.append(dist)

        # 3. Finger extension (tip y vs MCP y)
        #    If tip.y < mcp.y → finger is extended (pointing up)
        finger_mcps = [2, 5, 9, 13, 17]
        for tip, mcp in zip(FINGER_TIPS, finger_mcps):
            features.append(lm[tip][1] - lm[mcp][1])  # y difference

        # 4. Inter-finger distances (spread detection)
        for i in range(len(FINGER_TIPS) - 1):
            dist = np.linalg.norm(lm[FINGER_TIPS[i]] - lm[FINGER_TIPS[i+1]])
            features.append(dist)

        return features

    def load_gestures(self):
        if os.path.exists("clean_gestures.pkl"):
            with open("clean_gestures.pkl", "rb") as f:
                self.gestures = pickle.load(f)

    def save_gestures(self):
        with open("clean_gestures.pkl", "wb") as f:
            pickle.dump(self.gestures, f)

    def remove_background_white(self, frame, result):
        if not result.hand_landmarks:
            return np.full_like(frame, (255, 255, 255), dtype=np.uint8)
        h, w = frame.shape[:2]
        mask   = np.zeros((h, w), dtype=np.uint8)
        points = np.array([[int(lm.x * w), int(lm.y * h)]
                           for lm in result.hand_landmarks[0]], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)
        mask     = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
        white_bg = np.full_like(frame, (255, 255, 255), dtype=np.uint8)
        mask3    = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.where(mask3 == 255, frame, white_bg)

    # ── Draw finger landmarks with colours ────────────────────
    def draw_landmarks(self, display, landmarks):
        colors = {
            'thumb':  (255,  0,   0),
            'index':  (  0, 255,  0),
            'middle': (  0,   0, 255),
            'ring':   (255, 255,  0),
            'pinky':  (255,   0, 255),
            'wrist':  (  0, 255, 255),
        }
        finger_ranges = [(0,0),(1,4),(5,8),(9,12),(13,16),(17,20)]
        finger_names  = ['wrist','thumb','index','middle','ring','pinky']

        for i, pt in enumerate(landmarks):
            x = int(pt[0] * 640)
            y = int(pt[1] * 480)
            color = (255, 0, 0)
            for fi, (start, end) in enumerate(finger_ranges):
                if start <= i <= end:
                    color = colors[finger_names[fi]]
                    break
            cv2.circle(display, (x, y), 5, color, -1)

        # Draw connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        for a, b in connections:
            x1, y1 = int(landmarks[a][0]*640), int(landmarks[a][1]*480)
            x2, y2 = int(landmarks[b][0]*640), int(landmarks[b][1]*480)
            cv2.line(display, (x1,y1), (x2,y2), (200,200,200), 1)

if __name__ == "__main__":
    system = DSPGestureTrainer()
    cap    = cv2.VideoCapture(0)

    print("\n╔══════════════════════════════════════╗")
    print("║       DSP GESTURE TRAINER            ║")
    print("╠══════════════════════════════════════╣")
    print("║  R = Start/Stop Recording            ║")
    print("║  S = Save gesture                    ║")
    print("║  Q = Quit                            ║")
    print("╚══════════════════════════════════════╝\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result  = system.detector.detect(mp_image)
        display = system.remove_background_white(frame, result)

        if result.hand_landmarks:
            raw    = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
            smooth = system.apply_fir_filter(raw)

            # Extract full finger features
            features = system.extract_finger_features(smooth)

            if system.is_recording:
                system.current_recording.append(features)
                system.frames_for_gif.append(display.copy())

            # Draw coloured finger landmarks
            system.draw_landmarks(display, smooth)

            # Show finger curl status on screen
            lm = np.array(smooth)
            finger_names = ['Thumb','Index','Middle','Ring','Pinky']
            for i, (tip, base) in enumerate(zip(FINGER_TIPS, FINGER_BASES)):
                dist      = np.linalg.norm(lm[tip] - lm[base])
                state     = "Open" if dist > 0.15 else "Curl"
                color     = (0, 200, 0) if state == "Open" else (0, 0, 255)
                cv2.putText(display, f"{finger_names[i]}: {state}",
                            (10, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2)

        # Recording indicator
        if system.is_recording:
            cv2.putText(display, "● RECORDING", (430, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, f"Frames: {len(system.current_recording)}",
                        (430, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1)

        # Gesture count
        cv2.putText(display, f"Gestures saved: {len(system.gestures)}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        cv2.imshow("DSP Trainer", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            system.is_recording = not system.is_recording
            if system.is_recording:
                system.current_recording, system.frames_for_gif = [], []
                print("▶ Recording started...")
            else:
                print(f"⏹ Recording stopped. ({len(system.current_recording)} frames)")
        elif key == ord('s') and system.current_recording:
            name = input("Enter Gesture Name: ").strip().lower()
            if name:
                if name not in system.gestures:
                    system.gestures[name] = []
                system.gestures[name].append(system.current_recording[:])
                system.save_gestures()
                imageio.mimsave(f"{name}.gif",
                    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in system.frames_for_gif[::2]],
                    fps=15)
                print(f"✅ Saved: '{name}' ({len(system.gestures[name])} samples total)")

    cap.release()
    cv2.destroyAllWindows()