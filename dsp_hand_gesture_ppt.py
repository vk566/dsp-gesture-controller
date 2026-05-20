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
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.gestures         = self.load_gestures()
        self.current_seq      = []
        self.last_action_time = 0
        self.is_locked        = False

        # DSP: IIR Filter
        self.prev_smooth = None
        self.alpha       = 0.5

    def load_gestures(self):
        if os.path.exists("clean_gestures.pkl"):
            with open("clean_gestures.pkl", "rb") as f:
                return pickle.load(f)
        return {}

    def apply_iir(self, raw):
        raw = np.array(raw)
        if self.prev_smooth is None:
            self.prev_smooth = raw
            return raw.tolist()
        smoothed         = (self.alpha * raw) + ((1 - self.alpha) * self.prev_smooth)
        self.prev_smooth = smoothed
        return smoothed.tolist()

    def extract_finger_features(self, landmarks):
        lm = np.array(landmarks)
        features = []
        features.extend(lm.flatten().tolist())
        for tip, base in zip(FINGER_TIPS, FINGER_BASES):
            features.append(np.linalg.norm(lm[tip] - lm[base]))
        finger_mcps = [2, 5, 9, 13, 17]
        for tip, mcp in zip(FINGER_TIPS, finger_mcps):
            features.append(lm[tip][1] - lm[mcp][1])
        for i in range(len(FINGER_TIPS) - 1):
            features.append(np.linalg.norm(lm[FINGER_TIPS[i]] - lm[FINGER_TIPS[i+1]]))
        return features

    def recognize_gesture(self):
        if not self.gestures or len(self.current_seq) < 20:
            return "None"

        y_start      = np.array(self.current_seq[0])[:21*3].reshape(21,3)[0][1]
        y_end        = np.array(self.current_seq[-1])[:21*3].reshape(21,3)[0][1]
        total_y_move = abs(y_end - y_start)
        is_moving_up = y_end < y_start

        best_score = float('inf')
        best_name  = "None"

        for name, sequences in self.gestures.items():
            for s in sequences:
                ml    = min(len(self.current_seq), len(s))
                score = np.mean([
                    np.linalg.norm(np.array(self.current_seq[-ml:][i]) - np.array(s[-ml:][i]))
                    for i in range(ml)
                ])
                if score < best_score:
                    best_score = score
                    best_name  = name

        if "next" in best_name:
            if not is_moving_up or total_y_move < 0.05: return "None"
        if "previous" in best_name:
            if is_moving_up or total_y_move < 0.05: return "None"

        return best_name

    def execute_command(self, name):
        n = name.lower()
        if "next" in n:
            pyautogui.press('right')
            print(">>> [NEXT SLIDE]")
        elif "previous" in n:
            pyautogui.press('left')
            print("<<< [PREVIOUS SLIDE]")
        elif "start" in n:
            pyautogui.press('f5')
            print("▶ [START SLIDESHOW]")
        elif "exit" in n:
            pyautogui.press('esc')
            print("⏹ [EXIT SLIDESHOW]")

    def run(self):
        cap = cv2.VideoCapture(0)
        print("\n✅ DSP CONTROLLER ACTIVE (Running in background)")
        print("   Camera is ON but hidden — gesture away!")
        print("   Press CTRL+C in this window to stop.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame    = cv2.flip(frame, 1)
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = self.detector.detect(mp_image)

                if result.hand_landmarks:
                    raw      = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
                    smooth   = self.apply_iir(raw)
                    features = self.extract_finger_features(smooth)
                    self.current_seq.append(features)
                    if len(self.current_seq) > 25:
                        self.current_seq.pop(0)

                    gesture = self.recognize_gesture()

                    if gesture != "None" and not self.is_locked:
                        self.execute_command(gesture)
                        self.is_locked        = True
                        self.last_action_time = time.time()

                    if self.is_locked and (time.time() - self.last_action_time > 1.2):
                        self.is_locked = False
                else:
                    self.prev_smooth = None
                    self.is_locked   = False

                # ── NO cv2.imshow — runs silently in background ──
                # Just check for key press to quit (optional)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n⏹ Controller stopped.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    EasyDSPController().run()