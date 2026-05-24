import cv2
import mediapipe as mp
import numpy as np
import pickle
import imageio
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5,  9, 13, 17]

class DSPGestureTrainer:
    def __init__(self):
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.gestures          = {}
        self.is_recording      = False
        self.current_recording = []
        self.frames_for_gif    = []
        self.fir_buffers       = {}
        self.fir_window        = 3

        # ── Air Draw Lock ─────────────────────────────────────
        self.is_drawing_key  = False
        self.air_draw_path   = []
        self.fir_draw_buffer = []
        self.last_point      = None  # to avoid duplicate points

        self.load_gestures()

    def apply_fir_filter(self, landmarks, hand_key):
        if hand_key not in self.fir_buffers:
            self.fir_buffers[hand_key] = []
        buf = self.fir_buffers[hand_key]
        buf.append(landmarks)
        if len(buf) > self.fir_window:
            buf.pop(0)
        return np.mean(buf, axis=0).tolist()

    # ── FIR for drawing path ──────────────────────────────────
    # y[n] = (x[n] + x[n-1] + x[n-2]) / 3
    def apply_fir_to_point(self, point):
        self.fir_draw_buffer.append(point)
        if len(self.fir_draw_buffer) > self.fir_window:
            self.fir_draw_buffer.pop(0)
        arr = np.array(self.fir_draw_buffer)
        return tuple(np.mean(arr, axis=0).tolist())

    def get_hand_orientation(self, landmarks):
        lm      = np.array(landmarks)
        v1      = lm[5]  - lm[0]
        v2      = lm[17] - lm[0]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        return 1.0 if cross_z > 0 else 0.0

    # ── Check if index finger is pointing (extended) ──────────
    # Index tip (8) must be extended, other fingers curled
    def is_index_pointing(self, landmarks):
        lm = np.array(landmarks)
        # Index finger extended: tip higher than base
        index_extended = np.linalg.norm(lm[8] - lm[5]) > 0.08
        # Middle, ring, pinky must be curled
        middle_curled  = np.linalg.norm(lm[12] - lm[9])  < 0.08
        ring_curled    = np.linalg.norm(lm[16] - lm[13]) < 0.08
        pinky_curled   = np.linalg.norm(lm[20] - lm[17]) < 0.08
        return index_extended and middle_curled and ring_curled and pinky_curled

    def extract_features(self, landmarks, hand_label):
        lm = np.array(landmarks)
        features = []
        features.extend(lm.flatten().tolist())
        for tip, base in zip(FINGER_TIPS, FINGER_BASES):
            features.append(np.linalg.norm(lm[tip] - lm[base]))
        for tip, mcp in zip(FINGER_TIPS, [2,5,9,13,17]):
            features.append(lm[tip][1] - lm[mcp][1])
        for i in range(len(FINGER_TIPS)-1):
            features.append(np.linalg.norm(lm[FINGER_TIPS[i]] - lm[FINGER_TIPS[i+1]]))
        features.append(self.get_hand_orientation(landmarks))
        features.append(1.0 if hand_label == "Right" else 0.0)
        return features

    def load_gestures(self):
        if os.path.exists("clean_gestures.pkl"):
            with open("clean_gestures.pkl", "rb") as f:
                self.gestures = pickle.load(f)

    def save_gestures(self):
        with open("clean_gestures.pkl", "wb") as f:
            pickle.dump(self.gestures, f)

    def save_lock_key(self, path):
        path_arr = np.array(path)
        # Normalize
        path_arr[:,0] = (path_arr[:,0] - path_arr[:,0].min()) / (path_arr[:,0].max() - path_arr[:,0].min() + 1e-6)
        path_arr[:,1] = (path_arr[:,1] - path_arr[:,1].min()) / (path_arr[:,1].max() - path_arr[:,1].min() + 1e-6)
        # Resample to 50 points
        indices = np.linspace(0, len(path_arr)-1, 50).astype(int)
        sampled = path_arr[indices]
        with open("lock_key.pkl", "wb") as f:
            pickle.dump(sampled.tolist(), f)
        print("🔑 Lock key saved!")

    def remove_background_white(self, frame, landmarks_list):
        display = np.full_like(frame, (255,255,255), dtype=np.uint8)
        h, w    = frame.shape[:2]
        for lms in landmarks_list:
            mask   = np.zeros((h,w), dtype=np.uint8)
            points = np.array([[int(lm.x*w), int(lm.y*h)] for lm in lms], dtype=np.int32)
            cv2.fillConvexPoly(mask, points, 255)
            mask   = cv2.dilate(mask, np.ones((15,15), np.uint8), iterations=1)
            mask3  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            display = np.where(mask3==255, frame, display)
        return display

    def draw_landmarks(self, display, landmarks):
        finger_colors = [(0,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
        ranges = [(0,0),(1,4),(5,8),(9,12),(13,16),(17,20)]
        for i, pt in enumerate(landmarks):
            x, y  = int(pt[0]*640), int(pt[1]*480)
            color = finger_colors[0]
            for fi,(s,e) in enumerate(ranges):
                if s <= i <= e: color = finger_colors[fi]; break
            cv2.circle(display, (x,y), 5, color, -1)
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
        ]
        for a,b in connections:
            x1,y1 = int(landmarks[a][0]*640), int(landmarks[a][1]*480)
            x2,y2 = int(landmarks[b][0]*640), int(landmarks[b][1]*480)
            cv2.line(display,(x1,y1),(x2,y2),(180,180,180),1)


if __name__ == "__main__":
    system = DSPGestureTrainer()
    cap    = cv2.VideoCapture(0)

    print("\n╔══════════════════════════════════════════╗")
    print("║         DSP GESTURE TRAINER              ║")
    print("╠══════════════════════════════════════════╣")
    print("║  R  = Start/Stop Recording gesture       ║")
    print("║  S  = Save gesture                       ║")
    print("║  K  = Record secret air-draw lock key    ║")
    print("║  Q  = Quit                               ║")
    print("╠══════════════════════════════════════════╣")
    print("║  HOW TO DRAW SECRET KEY:                 ║")
    print("║  1. Press K to start                     ║")
    print("║  2. Point ONLY index finger              ║")
    print("║  3. Draw your pattern in air             ║")
    print("║  4. Curl all fingers to STOP drawing     ║")
    print("║  5. Press K to save                      ║")
    print("╚══════════════════════════════════════════╝\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = system.detector.detect(mp_image)

        all_lms = result.hand_landmarks if result.hand_landmarks else []
        display = system.remove_background_white(frame, all_lms)

        frame_features = []

        for idx, hand_lms in enumerate(result.hand_landmarks):
            hand_label = "Right"
            if result.handedness and idx < len(result.handedness):
                hand_label = result.handedness[idx][0].display_name

            raw      = [[lm.x, lm.y, lm.z] for lm in hand_lms]
            smooth   = system.apply_fir_filter(raw, hand_label)
            features = system.extract_features(smooth, hand_label)
            frame_features.extend(features)
            system.draw_landmarks(display, smooth)

            orient       = system.get_hand_orientation(smooth)
            orient_text  = "FRONT" if orient == 1.0 else "BACK"
            orient_color = (0,200,0) if orient == 1.0 else (0,100,255)
            wrist_x = int(smooth[0][0]*640)
            wrist_y = int(smooth[0][1]*480)
            cv2.putText(display, f"{hand_label} | {orient_text}",
                        (wrist_x-40, wrist_y+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, orient_color, 2)

            if idx == 0:
                lm           = np.array(smooth)
                finger_names = ['Thumb','Index','Middle','Ring','Pinky']
                for i,(tip,base) in enumerate(zip(FINGER_TIPS,FINGER_BASES)):
                    dist  = np.linalg.norm(lm[tip] - lm[base])
                    state = "Open" if dist > 0.15 else "Curl"
                    color = (0,200,0) if state == "Open" else (0,0,255)
                    cv2.putText(display, f"{finger_names[i]}: {state}",
                                (10, 30+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ── Air draw: ONLY when index is pointing ─────────
            if system.is_drawing_key:
                pointing = system.is_index_pointing(smooth)

                if pointing:
                    tip_x     = smooth[8][0]
                    tip_y     = smooth[8][1]
                    # FIR smooth: y[n] = (x[n] + x[n-1] + x[n-2]) / 3
                    smooth_pt = system.apply_fir_to_point((tip_x, tip_y))

                    # Only add if moved enough (avoid duplicate points)
                    if system.last_point is None or \
                       np.linalg.norm(np.array(smooth_pt) - np.array(system.last_point)) > 0.01:
                        system.air_draw_path.append(smooth_pt)
                        system.last_point = smooth_pt

                    # Draw path on screen
                    for pi in range(1, len(system.air_draw_path)):
                        p1 = (int(system.air_draw_path[pi-1][0]*640),
                              int(system.air_draw_path[pi-1][1]*480))
                        p2 = (int(system.air_draw_path[pi][0]*640),
                              int(system.air_draw_path[pi][1]*480))
                        cv2.line(display, p1, p2, (0,0,255), 3)

                    # Show fingertip dot in red
                    cv2.circle(display,
                               (int(smooth[8][0]*640), int(smooth[8][1]*480)),
                               10, (0,0,255), -1)

                    # Show drawing status
                    cv2.putText(display, "✏ DRAWING...",
                                (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                else:
                    # Index not pointing — show pause status
                    cv2.putText(display, "✋ CURL TO PAUSE",
                                (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,150,255), 2)

        if frame_features and system.is_recording:
            system.current_recording.append(frame_features)
            system.frames_for_gif.append(display.copy())

        # Recording indicator
        if system.is_recording:
            cv2.putText(display, "● RECORDING GESTURE", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(display, f"Frames: {len(system.current_recording)}",
                        (300,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1)

        # Air draw mode indicator
        if system.is_drawing_key:
            cv2.putText(display, f"KEY MODE ON | Points: {len(system.air_draw_path)}",
                        (150, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)
            cv2.putText(display, "Point index finger to draw | Press K to save",
                        (60, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,0), 1)

        # Lock key status
        lock_exists = os.path.exists("lock_key.pkl")
        lock_color  = (0,200,0) if lock_exists else (0,0,200)
        lock_text   = "Lock Key: SET ✓" if lock_exists else "Lock Key: NOT SET — Press K"
        cv2.putText(display, lock_text,
                    (10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lock_color, 1)
        cv2.putText(display, f"Gestures: {len(system.gestures)}",
                    (480,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        cv2.imshow("DSP Trainer", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            system.is_recording = not system.is_recording
            if system.is_recording:
                system.current_recording, system.frames_for_gif = [], []
                print("▶ Recording gesture started...")
            else:
                print(f"⏹ Stopped. ({len(system.current_recording)} frames)")
        elif key == ord('s') and system.current_recording:
            name = input("Enter Gesture Name: ").strip().lower()
            if name:
                if name not in system.gestures:
                    system.gestures[name] = []
                system.gestures[name].append(system.current_recording[:])
                system.save_gestures()
                imageio.mimsave(f"{name}.gif",
                    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in system.frames_for_gif[::2]], fps=15)
                print(f"✅ Saved: '{name}' ({len(system.gestures[name])} samples)")
        elif key == ord('k'):
            system.is_drawing_key = not system.is_drawing_key
            if system.is_drawing_key:
                system.air_draw_path   = []
                system.fir_draw_buffer = []
                system.last_point      = None
                print("✏ Draw mode ON — point index finger and draw your secret pattern!")
                print("   Curl fingers to pause drawing. Press K again to save.")
            else:
                if len(system.air_draw_path) > 10:
                    system.save_lock_key(system.air_draw_path)
                else:
                    print("⚠ Path too short! Try again.")

    cap.release()
    cv2.destroyAllWindows()