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
        self.detector         = vision.HandLandmarker.create_from_options(options)
        self.gestures         = self.load_gestures()
        self.biolock          = self.load_biolock()

        # ── Lock state ────────────────────────────────────────
        self.is_unlocked      = False if self.biolock else True
        self.unlock_time      = None
        self.AUTO_LOCK_SECS   = 300  # 5 min idle auto lock

        # ── Bio check buffer ──────────────────────────────────
        # Collect frames and check simultaneously
        self.bio_check_frames  = []
        self.BIO_CHECK_COUNT   = 60   # check over 2 seconds
        self.BIO_THRESHOLD     = 0.15 # how similar hand must be
        self.check_result_msg  = ""
        self.check_result_time = 0

        # ── Gesture control ───────────────────────────────────
        self.current_seq      = []
        self.last_action_time = 0
        self.is_locked_action = False
        self.stable_gesture   = "None"
        self.stable_count     = 0
        self.STABLE_FRAMES    = 8

        self.iir_prev = {}
        self.alpha    = 0.5

    def load_gestures(self):
        if os.path.exists("clean_gestures.pkl"):
            with open("clean_gestures.pkl", "rb") as f:
                return pickle.load(f)
        return {}

    def load_biolock(self):
        if os.path.exists("biolock.pkl"):
            with open("biolock.pkl", "rb") as f:
                return pickle.load(f)
        return None

    # ── IIR filter for gesture landmarks ─────────────────────
    # Formula: y[n] = α*x[n] + (1-α)*y[n-1]
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
        if hand_label == "Right":
            return 1.0 if cross_z > 0 else 0.0
        else:
            return 1.0 if cross_z < 0 else 0.0

    # ── Extract biometrics (same as trainer) ─────────────────
    def extract_biometrics(self, landmarks):
        lm         = np.array(landmarks)
        palm_width = np.linalg.norm(lm[5] - lm[17]) + 1e-6
        mcp_pts    = [2, 5, 9, 13, 17]
        finger_lengths = []
        for tip, mcp in zip(FINGER_TIPS, mcp_pts):
            finger_lengths.append(np.linalg.norm(lm[tip] - lm[mcp]) / palm_width)
        middle_len = finger_lengths[2] + 1e-6
        ratios     = [fl / middle_len for fl in finger_lengths]
        span       = np.linalg.norm(lm[4] - lm[20]) / palm_width
        knuckle    = np.linalg.norm(lm[5] - lm[9]) / palm_width
        return ratios + [span, knuckle]

    # ── Extract gesture features ──────────────────────────────
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
        features.append(self.get_hand_orientation(landmarks, hand_label))
        features.append(1.0 if hand_label == "Right" else 0.0)
        return features

    # ── Check biometric match ─────────────────────────────────
    # Compare saved hand measurements vs live hand
    # Uses FIR averaging over BIO_CHECK_COUNT frames
    def check_biometric(self, bio_frames):
        if not self.biolock or len(bio_frames) < 10:
            return False
        # FIR: average all collected frames
        # y = (1/N) * sum of all bio frames
        avg_live = np.mean(bio_frames, axis=0)
        saved    = np.array(self.biolock)
        # Euclidean distance between saved and live biometrics
        score    = np.linalg.norm(avg_live - saved)
        print(f"   Bio score: {score:.4f} (threshold: {self.BIO_THRESHOLD})")
        return score < self.BIO_THRESHOLD

    def recognize_gesture(self):
        if not self.gestures or len(self.current_seq) < 40:
            return "None"
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
        return best_name

    def execute_command(self, name):
        n = name.lower()
        if "next" in n:
            pyautogui.press('right')
            print(">>> [NEXT SLIDE]")
        elif "previous" in n or "prev" in n or "back" in n:
            pyautogui.press('left')
            print("<<< [PREVIOUS SLIDE]")
        elif "start" in n:
            pyautogui.press('f5')
            print("▶ [START SLIDESHOW]")
        elif "exit" in n or "stop" in n:
            pyautogui.press('esc')
            print("⏹ [EXIT SLIDESHOW]")
        self.unlock_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(0)

        if not self.biolock:
            print("\n⚠ No Bio-Lock found!")
            print("  Go to Trainer → press K to set your hand biometric.")
            print("  Running WITHOUT lock for now...\n")
        else:
            print("\n🔒 CONTROLLER LOCKED")
            print("   Show your hand to unlock!")
            print("   System checks hand size + gesture simultaneously.\n")

        print("   Press CTRL+C to stop.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame    = cv2.flip(frame, 1)
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = self.detector.detect(mp_image)

                # ── Auto lock ─────────────────────────────────
                if self.is_unlocked and self.unlock_time:
                    if time.time() - self.unlock_time > self.AUTO_LOCK_SECS:
                        self.is_unlocked       = False
                        self.unlock_time       = None
                        self.bio_check_frames  = []
                        print("\n🔒 Auto-locked after 5 min idle.")

                if result.hand_landmarks:
                    frame_features = []

                    for idx, hand_lms in enumerate(result.hand_landmarks):
                        hand_label = "Right"
                        if result.handedness and idx < len(result.handedness):
                            raw_label  = result.handedness[idx][0].display_name
                            hand_label = "Left" if raw_label == "Right" else "Right"

                        raw    = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                        smooth = self.apply_iir(raw, hand_label)

                        # ── LOCKED: collect biometrics ────────
                        if not self.is_unlocked and self.biolock:
                            bio = self.extract_biometrics(smooth)
                            self.bio_check_frames.append(bio)

                            # Check every BIO_CHECK_COUNT frames simultaneously
                            if len(self.bio_check_frames) >= self.BIO_CHECK_COUNT:
                                if self.check_biometric(self.bio_check_frames):
                                    self.is_unlocked      = True
                                    self.unlock_time      = time.time()
                                    self.bio_check_frames = []
                                    self.current_seq      = []
                                    self.check_result_msg  = "UNLOCKED!"
                                    self.check_result_time = time.time()
                                    print("✅ UNLOCKED! Hand biometric matched!")
                                else:
                                    self.bio_check_frames = []
                                    self.check_result_msg  = "NOT MATCHED! Try again."
                                    self.check_result_time = time.time()
                                    print("❌ Hand not recognized! Try again.")

                        # ── UNLOCKED: recognize gestures ──────
                        if self.is_unlocked:
                            features = self.extract_features(smooth, hand_label)
                            frame_features.extend(features)

                    # Gesture recognition
                    if self.is_unlocked and frame_features:
                        self.current_seq.append(frame_features)
                        if len(self.current_seq) > 40:
                            self.current_seq.pop(0)

                        if not self.is_locked_action:
                            gesture = self.recognize_gesture()
                            if gesture == self.stable_gesture and gesture != "None":
                                self.stable_count += 1
                            else:
                                self.stable_gesture = gesture
                                self.stable_count   = 0
                            if self.stable_count >= self.STABLE_FRAMES:
                                self.execute_command(gesture)
                                self.is_locked_action = True
                                self.last_action_time = time.time()
                                self.current_seq      = []
                                self.stable_count     = 0
                                self.stable_gesture   = "None"

                        if self.is_locked_action and (time.time() - self.last_action_time > 2.5):
                            self.is_locked_action = False

                else:
                    # Hand left frame — reset
                    for k in self.iir_prev:
                        self.iir_prev[k] = None
                    if not self.is_unlocked:
                        self.bio_check_frames = []
                    if self.is_unlocked:
                        self.is_locked_action = False
                        self.current_seq      = []
                        self.stable_count     = 0
                        self.stable_gesture   = "None"

                # ── Print status ──────────────────────────────
                if not self.is_unlocked and self.biolock:
                    frames_done = len(self.bio_check_frames)
                    pct         = int((frames_done / self.BIO_CHECK_COUNT) * 100)
                    print(f"\r   Scanning hand... {pct}%    ", end="")

                # Show check result for 2 seconds
                if self.check_result_msg and (time.time() - self.check_result_time < 2):
                    print(f"\r   {self.check_result_msg}    ", end="")

                cv2.waitKey(1)

        except KeyboardInterrupt:
            print("\n⏹ Controller stopped.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    EasyDSPController().run()