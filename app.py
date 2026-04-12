from flask import Flask, Response, render_template, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter

from translator import translate_sentence
from tts import speak

# ---------------- GLOBAL VARIABLES ----------------
currentword = ""
final_sentence = ""

prediction_buffer = []
hand_present = False  # IMPORTANT

# Flask app
app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = load_model("addedtrail1-atozspacedelete.h5")

# Label mapping
gesture_labels = [
    "A","B","C","D","DELETE","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","SPACE","T","U","V","W","X","Y","Z"
]

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- PARAMETERS ----------------
sequence_length = 30
frame_buffer = deque(maxlen=sequence_length)

# Webcam
cap = cv2.VideoCapture(0)

# ---------------- FRAME GENERATOR ----------------
def generate_frames():
    global prediction_buffer, currentword, final_sentence, hand_present

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # ================= HAND DETECTED =================
        if results.multi_hand_landmarks:

            # NEW gesture started
            if not hand_present:
                prediction_buffer.clear()
                hand_present = True

            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # -------- FEATURE EXTRACTION --------
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            landmarks = landmarks - landmarks[0]
            scale = np.linalg.norm(landmarks)
            if scale > 0:
                landmarks = landmarks / scale

            dist_8_12 = np.linalg.norm(landmarks[8] - landmarks[12])

            features = landmarks.flatten().tolist()
            features.append(dist_8_12)

            frame_buffer.append(features)

            if len(frame_buffer) == sequence_length:
                input_sequence = np.array(frame_buffer).reshape(1, sequence_length, 64)
                prediction = model.predict(input_sequence, verbose=0)[0]

                confidence = np.max(prediction)
                predicted_class = np.argmax(prediction)
                predicted_label = gesture_labels[predicted_class]

                if confidence > 0.8:
                    prediction_buffer.append(predicted_label)

                # Show live prediction
                cv2.putText(
                    frame,
                    f"Gesture: {predicted_label}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

        # ================= HAND REMOVED =================
        else:
            if hand_present:
                # Gesture finished → finalize letter

                if prediction_buffer:
                    most_common, count = Counter(prediction_buffer).most_common(1)[0]

                    if count >= 5:  # threshold to remove noise
    # -------- HANDLE SPECIAL SIGNS --------
                        if most_common == "SPACE":
                            currentword += " "

                        elif most_common == "DELETE":
                            currentword = currentword[:-1]  # remove last char

                        else:
                            currentword += most_common

    # Update final sentence
                        final_sentence = currentword
                        print("Final Letter:", most_common)

                        

                # Reset for next gesture
                prediction_buffer.clear()
                hand_present = False

        # -------- STREAM FRAME --------
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/predict")
def predict():
    return jsonify({
        "sentence": final_sentence
    })

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data["text"]

    translations = translate_sentence(text)

    return jsonify(translations)

@app.route("/speak", methods=["POST"])
def speak_route():
    data = request.get_json()
    text = data["text"]
    speak(text)
    return jsonify({"status": "spoken"})

@app.route("/reset")
def reset():
    global currentword, final_sentence, prediction_buffer

    currentword = ""
    final_sentence = ""
    prediction_buffer.clear()

    return jsonify({"status": "reset"})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True) 
    