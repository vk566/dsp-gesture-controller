import cv2
import mediapipe as mp
import numpy as np
import pickle
import imageio
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DSPGestureTrainer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'hand_landmarker.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.gestures = {}
        self.is_recording = False
        self.current_recording = []
        self.frames_for_gif = []
        
        # DSP: FIR Filter (Moving Average)
        self.fir_buffer = []
        self.fir_window = 3
        
        self.load_gestures()

    def apply_fir_filter(self, landmarks):
        self.fir_buffer.append(landmarks)
        if len(self.fir_buffer) > self.fir_window:
            self.fir_buffer.pop(0)
        return np.mean(self.fir_buffer, axis=0).tolist()

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
        mask = np.zeros((h, w), dtype=np.uint8)
        points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in result.hand_landmarks[0]], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)
        mask = cv2.dilate(mask, np.ones((15,15), np.uint8), iterations=1)
        white_bg = np.full_like(frame, (255, 255, 255), dtype=np.uint8)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.where(mask3 == 255, frame, white_bg)

if __name__ == "__main__":
    system = DSPGestureTrainer()
    cap = cv2.VideoCapture(0)
    print("\nDSP TRAINER: R=Record, S=Save, D=Delete, Q=Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = system.detector.detect(mp_image)
        display = system.remove_background_white(frame, result)

        if result.hand_landmarks:
            raw = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
            smooth = system.apply_fir_filter(raw)
            if system.is_recording:
                system.current_recording.append(smooth)
                system.frames_for_gif.append(display.copy())
            for pt in smooth:
                cv2.circle(display, (int(pt[0]*640), int(pt[1]*480)), 4, (255, 0, 0), -1)

        cv2.imshow("DSP Trainer", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            system.is_recording = not system.is_recording
            if system.is_recording: 
                system.current_recording, system.frames_for_gif = [], []
                print("Recording started...")
            else: print("Recording stopped.")
        elif key == ord('s') and system.current_recording:
            name = input("Enter Gesture Name: ").strip().lower()
            if name:
                if name not in system.gestures: system.gestures[name] = []
                system.gestures[name].append(system.current_recording[:])
                system.save_gestures()
                imageio.mimsave(f"{name}.gif", [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in system.frames_for_gif[::2]], fps=15)
                print(f"Saved: {name}")

    cap.release()
    cv2.destroyAllWindows()