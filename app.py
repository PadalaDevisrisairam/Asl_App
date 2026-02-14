from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
sentence_buffer = []
last_word = None
app = Flask(__name__)

# Load trained Bi-LSTM model
model = load_model("dynamic_sign_bilstm_model_9.h5")

# Label mapping (must match training order)
gesture_labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Parameters
sequence_length = 30
frame_buffer = deque(maxlen=sequence_length)

# Webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # -------- FEATURE EXTRACTION (64 FEATURES) --------
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Normalize (wrist origin + scale)
            landmarks = landmarks - landmarks[0]
            scale = np.linalg.norm(landmarks)
            if scale > 0:
                landmarks = landmarks / scale

            # Finger distance (index 8, middle 12)
            dist_8_12 = np.linalg.norm(landmarks[8] - landmarks[12])

            features = landmarks.flatten().tolist()
            features.append(dist_8_12)

            frame_buffer.append(features)

            if len(frame_buffer) == sequence_length:
                input_sequence = np.array(frame_buffer).reshape(1, sequence_length, 64)
                prediction = model.predict(input_sequence, verbose=0)[0]
                predicted_class = np.argmax(prediction)
                predicted_label = gesture_labels[predicted_class]

                cv2.putText(
                    frame,
                    f"Gesture: {predicted_label}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
