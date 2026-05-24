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

# ── DTW Algorithm ─────────────────────────────────────────────
# Formula: DTW(i,j) = dist(i,j) + min(DTW(i-1,j), DTW(i,j-1), DTW(i-1,j-1))
def dtw_distance(path1, path2):
    p1 = np.array(path1)
    p2 = np.array(path2)
    n, m = len(p1), len(p2)
    # Initialize DTW matrix with infinity
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Euclidean distance between two points
            dist = np.linalg.norm(p1[i-1] - p2[j-1])
            # DTW recurrence formula
            dtw[i,j] = dist + min(
                dtw[i-1, j],     # insertion
                dtw[i, j-1],     # deletion
                dtw[i-1, j-1]    # match
            )
    return dtw[n, m]

# ── Normalize + resample path to 50 points ───────────────────
def normalize_path(path):
    arr = np.array(path)
    arr[:,0] = (arr[:,0] - arr[:,0].min()) / (arr[:,0].max() - arr[:,0].min() + 1e-6)
    arr[:,1] = (arr[:,1] - arr[:,1].min()) / (arr[:,1].max() - arr[:,1].min() + 1e-6)
    indices  = np.linspace(0, len(arr)-1, 50).astype(int)
    return arr[indices].tolist()

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
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.gestures         = self.load_gestures()
        self.lock_key         = self.load_lock_key()

        # ── Lock state ────────────────────────────────────────
        self.is_unlocked      = False
        self.unlock_time      = None
        self.AUTO_LOCK_SECS   = 300   # auto lock after 5 minutes idle

        # ── Air draw for unlock ───────────────────────────────
        self.is_drawing       = False
        self.live_path        = []
        self.iir_draw_prev    = None  # IIR previous value for drawing
        self.alpha_draw       = 0.5   # IIR alpha for drawing

        # ── Gesture control ───────────────────────────────────
        self.current_seq      = []
        self.last_action_time = 0
        self.is_locked_action = False
        self.stable_gesture   = "None"
        self.stable_count     = 0
        self.STABLE_FRAMES    = 8

        self.iir_prev = {}
        self.alpha    = 0.5

        self.DTW_THRESHOLD = 0.25   # lower = stricter match

    def load_gestures(self):
        if os.path.exists("clean_gestures.pkl"):
            with open("clean_gestures.pkl", "rb") as f:
                return pickle.load(f)
        return {}

    def load_lock_key(self):
        if os.path.exists("lock_key.pkl"):
            with open("lock_key.pkl", "rb") as f:
                return pickle.load(f)
        return None

    # ── IIR for gesture landmarks ─────────────────────────────
    # Formula: y[n] = α*x[n] + (1-α)*y[n-1]
    def apply_iir(self, raw, hand_key):
        raw = np.array(raw)
        if hand_key not in self.iir_prev or self.iir_prev[hand_key] is None:
            self.iir_prev[hand_key] = raw
            return raw.tolist()
        smoothed = (self.alpha * raw) + ((1 - self.alpha) * self.iir_prev[hand_key])
        self.iir_prev[hand_key] = smoothed
        return smoothed.tolist()

    # ── IIR for live air drawing ──────────────────────────────
    # Formula: y[n] = α*x[n] + (1-α)*y[n-1]
    def apply_iir_draw(self, point):
        pt = np.array(point)
        if self.iir_draw_prev is None:
            self.iir_draw_prev = pt
            return tuple(pt.tolist())
        smoothed           = (self.alpha_draw * pt) + ((1 - self.alpha_draw) * self.iir_draw_prev)
        self.iir_draw_prev = smoothed
        return tuple(smoothed.tolist())

    def get_hand_orientation(self, landmarks):
        lm      = np.array(landmarks)
        v1      = lm[5]  - lm[0]
        v2      = lm[17] - lm[0]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        return 1.0 if cross_z > 0 else 0.0

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

    # ── Check if ONLY index finger is pointing ───────────────
    def is_index_pointing(self, landmarks):
        lm = np.array(landmarks)
        wrist = lm[0]
        index_dist  = np.linalg.norm(lm[8]  - wrist)
        middle_dist = np.linalg.norm(lm[12] - wrist)
        ring_dist   = np.linalg.norm(lm[16] - wrist)
        pinky_dist  = np.linalg.norm(lm[20] - wrist)
        index_extended  = index_dist > 0.22
        middle_curled   = middle_dist < 0.12
        ring_curled     = ring_dist   < 0.12
        pinky_curled    = pinky_dist  < 0.12
        index_dominant  = index_dist > (middle_dist * 2.0)
        return index_extended and middle_curled and ring_curled and pinky_curled and index_dominant

    # ── Check if live drawn path matches saved key ────────────
    def check_unlock(self):
        if not self.lock_key or len(self.live_path) < 10:
            return False
        # Normalize live path
        norm_live = normalize_path(self.live_path)
        # DTW comparison
        score = dtw_distance(norm_live, self.lock_key)
        # Normalize score by path length
        score = score / 50
        print(f"   DTW Score: {score:.4f} (threshold: {self.DTW_THRESHOLD})")
        return score < self.DTW_THRESHOLD

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
        # Reset auto-lock timer on every action
        self.unlock_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(0)

        if not self.lock_key:
            print("\n⚠ No lock key found!")
            print("  Go to Option 1 (Trainer) and press K to set your secret pattern.")
            print("  Running WITHOUT lock for now...\n")
            self.is_unlocked = True
        else:
            print("\n🔒 CONTROLLER LOCKED")
            print("   Draw your secret pattern in air to unlock!")
            print("   Point your INDEX finger and draw — lift hand to confirm.\n")

        print("   Press CTRL+C to stop.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame    = cv2.flip(frame, 1)
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = self.detector.detect(mp_image)

                # ── Auto lock after idle ──────────────────────
                if self.is_unlocked and self.unlock_time:
                    if time.time() - self.unlock_time > self.AUTO_LOCK_SECS:
                        self.is_unlocked    = False
                        self.unlock_time    = None
                        self.iir_draw_prev  = None
                        print("\n🔒 Auto-locked after 5 minutes idle.")

                if result.hand_landmarks:
                    frame_features = []

                    for idx, hand_lms in enumerate(result.hand_landmarks):
                        hand_label = "Right"
                        if result.handedness and idx < len(result.handedness):
                            hand_label = result.handedness[idx][0].display_name

                        raw    = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                        smooth = self.apply_iir(raw, hand_label)

                        # ── LOCKED: track index fingertip for air draw ──
                        if not self.is_unlocked:
                            tip_x = smooth[8][0]
                            tip_y = smooth[8][1]
                            # IIR smooth the drawing point
                            # y[n] = 0.5*x[n] + 0.5*y[n-1]
                            smooth_pt = self.apply_iir_draw((tip_x, tip_y))
                            self.live_path.append(smooth_pt)
                            self.is_drawing = True

                        # ── UNLOCKED: recognize gestures ────────────────
                        else:
                            features = self.extract_features(smooth, hand_label)
                            frame_features.extend(features)

                    # ── Gesture recognition (only when unlocked) ──────
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
                    # ── Hand left frame ────────────────────────
                    if not self.is_unlocked and self.is_drawing and len(self.live_path) > 10:
                        # Hand removed → check the drawn path
                        print("   Checking pattern...")
                        if self.check_unlock():
                            self.is_unlocked   = True
                            self.unlock_time   = time.time()
                            self.current_seq   = []
                            print("✅ UNLOCKED! You can now control the PPT!")
                        else:
                            print("❌ Pattern does not match! Try again.")

                    # Reset draw state
                    self.live_path     = []
                    self.is_drawing    = False
                    self.iir_draw_prev = None

                    # Reset IIR
                    for k in self.iir_prev: self.iir_prev[k] = None
                    if self.is_unlocked:
                        self.is_locked_action = False
                        self.current_seq      = []
                        self.stable_count     = 0
                        self.stable_gesture   = "None"

                cv2.waitKey(1)

        except KeyboardInterrupt:
            print("\n⏹ Controller stopped.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    EasyDSPController().run()