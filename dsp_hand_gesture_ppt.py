import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5,  9, 13, 17]

class EasyDSPController:
    def __init__(self):
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7
        )
        self.detector       = vision.HandLandmarker.create_from_options(options)
        self.gestures       = self.load_gestures()

        self.current_seq    = []
        self.action_locked  = False
        self.last_action_time = 0
        self.stable_gesture = "None"
        self.stable_count   = 0
        self.STABLE_FRAMES  = 8

        self.iir_prev = {}
        self.alpha    = 0.5

    def load_gestures(self):
        if os.path.exists("clean_gestures.pkl"):
            with open("clean_gestures.pkl", "rb") as f:
                return pickle.load(f)
        return {}

    def apply_iir(self, raw, hand_key):
        raw = np.array(raw)
        if hand_key not in self.iir_prev or self.iir_prev[hand_key] is None:
            self.iir_prev[hand_key] = raw
            return raw.tolist()
        smoothed = (self.alpha * raw) + ((1 - self.alpha) * self.iir_prev[hand_key])
        self.iir_prev[hand_key] = smoothed
        return smoothed.tolist()

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

    def recognize_gesture(self):
        if not self.gestures or len(self.current_seq) < 40:
            return "None"
        best_score = float('inf')
        best_name  = "None"
        for name, seqs in self.gestures.items():
            for s in seqs:
                ml    = min(len(self.current_seq), len(s))
                score = np.mean([
                    np.linalg.norm(np.array(self.current_seq[-ml:][i]) - np.array(s[-ml:][i]))
                    for i in range(ml)
                ])
                if score < best_score:
                    best_score = score
                    best_name  = name
        return best_name

    def execute_command(self, name):
        n = name.lower()
        if "next" in n:
            pyautogui.press('right')
            print("\n>>> [NEXT SLIDE]")
        elif "previous" in n or "prev" in n or "back" in n:
            pyautogui.press('left')
            print("\n<<< [PREVIOUS SLIDE]")
        elif "start" in n:
            pyautogui.press('f5')
            print("\n▶ [START SLIDESHOW]")
        elif "exit" in n or "stop" in n:
            pyautogui.press('esc')
            print("\n⏹ [EXIT SLIDESHOW]")

    def run(self):
        cap = cv2.VideoCapture(0)
        print("\n✅ DSP CONTROLLER ACTIVE")
        print(f"   Gestures loaded: {list(self.gestures.keys())}")
        print("   Press CTRL+C to stop.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame    = cv2.flip(frame, 1)
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = self.detector.detect(mp_image)

                if result.hand_landmarks:
                    frame_features = []

                    for idx, hand_lms in enumerate(result.hand_landmarks):
                        hand_label = "Right"
                        if result.handedness and idx < len(result.handedness):
                            raw_label  = result.handedness[idx][0].display_name
                            hand_label = "Left" if raw_label == "Right" else "Right"

                        raw    = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                        smooth = self.apply_iir(raw, hand_label)
                        frame_features.extend(self.extract_features(smooth, hand_label))

                    # Gesture recognition
                    self.current_seq.append(frame_features)
                    if len(self.current_seq) > 40:
                        self.current_seq.pop(0)

                    if not self.action_locked:
                        gesture = self.recognize_gesture()
                        if gesture == self.stable_gesture and gesture != "None":
                            self.stable_count += 1
                        else:
                            self.stable_gesture = gesture
                            self.stable_count   = 0

                        if self.stable_count >= self.STABLE_FRAMES:
                            self.execute_command(gesture)
                            self.action_locked    = True
                            self.last_action_time = time.time()
                            self.current_seq      = []
                            self.stable_count     = 0
                            self.stable_gesture   = "None"

                    if self.action_locked and (time.time() - self.last_action_time > 2.5):
                        self.action_locked = False

                else:
                    for k in self.iir_prev: self.iir_prev[k] = None
                    self.current_seq    = []
                    self.stable_count   = 0
                    self.stable_gesture = "None"
                    self.action_locked  = False

                cv2.waitKey(1)

        except KeyboardInterrupt:
            print("\n⏹ Controller stopped.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    EasyDSPController().run()