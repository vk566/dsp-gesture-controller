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
        self.detector          = vision.HandLandmarker.create_from_options(options)
        self.gestures          = {}
        self.is_recording      = False
        self.current_recording = []
        self.frames_for_gif    = []
        self.fir_buffers       = {}
        self.fir_window        = 3
        self.load_gestures()

    # FIR: y[n] = (x[n] + x[n-1] + x[n-2]) / 3
    def apply_fir_filter(self, landmarks, hand_key):
        if hand_key not in self.fir_buffers:
            self.fir_buffers[hand_key] = []
        buf = self.fir_buffers[hand_key]
        buf.append(landmarks)
        if len(buf) > self.fir_window:
            buf.pop(0)
        return np.mean(buf, axis=0).tolist()

    def get_hand_orientation(self, landmarks, hand_label):
        lm      = np.array(landmarks)
        v1      = lm[5]  - lm[0]
        v2      = lm[17] - lm[0]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        return (1.0 if cross_z > 0 else 0.0) if hand_label == "Right" else (1.0 if cross_z < 0 else 0.0)

    def extract_features(self, landmarks, hand_label):
        lm = np.array(landmarks)
        f  = []
        f.extend(lm.flatten().tolist())
        for tip, base in zip(FINGER_TIPS, FINGER_BASES):
            f.append(np.linalg.norm(lm[tip] - lm[base]))
        for tip, mcp in zip(FINGER_TIPS, [2,5,9,13,17]):
            f.append(lm[tip][1] - lm[mcp][1])
        for i in range(len(FINGER_TIPS)-1):
            f.append(np.linalg.norm(lm[FINGER_TIPS[i]] - lm[FINGER_TIPS[i+1]]))
        f.append(self.get_hand_orientation(landmarks, hand_label))
        f.append(1.0 if hand_label == "Right" else 0.0)
        return f

    def load_gestures(self):
        if os.path.exists("clean_gestures.pkl"):
            with open("clean_gestures.pkl", "rb") as f:
                self.gestures = pickle.load(f)

    def save_gestures(self):
        with open("clean_gestures.pkl", "wb") as f:
            pickle.dump(self.gestures, f)

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
        colors = [(0,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
        ranges = [(0,0),(1,4),(5,8),(9,12),(13,16),(17,20)]
        for i, pt in enumerate(landmarks):
            x, y  = int(pt[0]*640), int(pt[1]*480)
            color = colors[0]
            for fi,(s,e) in enumerate(ranges):
                if s <= i <= e: color = colors[fi]; break
            cv2.circle(display, (x,y), 5, color, -1)
        for a,b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
            cv2.line(display,
                     (int(landmarks[a][0]*640), int(landmarks[a][1]*480)),
                     (int(landmarks[b][0]*640), int(landmarks[b][1]*480)),
                     (180,180,180), 1)


if __name__ == "__main__":
    system = DSPGestureTrainer()
    cap    = cv2.VideoCapture(0)

    print("\n╔══════════════════════════════════════════╗")
    print("║         DSP GESTURE TRAINER              ║")
    print("╠══════════════════════════════════════════╣")
    print("║  R = Start/Stop Recording                ║")
    print("║  S = Save gesture                        ║")
    print("║  Q = Quit                                ║")
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
                raw_label  = result.handedness[idx][0].display_name
                hand_label = "Left" if raw_label == "Right" else "Right"

            raw      = [[lm.x, lm.y, lm.z] for lm in hand_lms]
            smooth   = system.apply_fir_filter(raw, hand_label)
            features = system.extract_features(smooth, hand_label)
            frame_features.extend(features)
            system.draw_landmarks(display, smooth)

            orient       = system.get_hand_orientation(smooth, hand_label)
            orient_text  = "FRONT" if orient == 1.0 else "BACK"
            orient_color = (0,200,0) if orient == 1.0 else (0,100,255)
            wrist_x      = int(smooth[0][0]*640)
            wrist_y      = int(smooth[0][1]*480)
            cv2.putText(display, f"{hand_label} | {orient_text}",
                        (wrist_x-40, wrist_y+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, orient_color, 2)

            if idx == 0:
                lm = np.array(smooth)
                for i,(tip,base) in enumerate(zip(FINGER_TIPS,FINGER_BASES)):
                    dist  = np.linalg.norm(lm[tip]-lm[base])
                    state = "Open" if dist > 0.15 else "Curl"
                    color = (0,200,0) if state == "Open" else (0,0,255)
                    cv2.putText(display,
                                f"{['Thumb','Index','Middle','Ring','Pinky'][i]}: {state}",
                                (10, 30+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if frame_features and system.is_recording:
            system.current_recording.append(frame_features)
            system.frames_for_gif.append(display.copy())

        if system.is_recording:
            cv2.putText(display, "● RECORDING", (450, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(display, f"Frames: {len(system.current_recording)}",
                        (450,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1)

        cv2.putText(display, f"Gestures: {list(system.gestures.keys())}",
                    (10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)

        cv2.imshow("DSP Trainer", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            system.is_recording = not system.is_recording
            if system.is_recording:
                system.current_recording = []
                system.frames_for_gif    = []
                print("▶ Recording started...")
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

    cap.release()
    cv2.destroyAllWindows()