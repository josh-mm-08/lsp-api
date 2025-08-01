from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from keras.models import load_model 
import cv2
import pickle
import tempfile
import os
import mediapipe as mp

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model1.h5')

# Load the LabelEncoder
with open('label_encoder1.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to extract landmakrs from frames of a video
def extract_keypoints_from_video(video_path):
    sequence = []
    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)

            if results.pose_landmarks and results.face_landmarks and results.left_hand_landmarks and results.right_hand_landmarks:
                pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
                face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten()
                lh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten()
                rh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten()

            else:
                pose = np.zeros(33*4)
                face = np.zeros(468*4)
                lh = np.zeros(21*4)
                rh = np.zeros(21*4)
                
            keypoints = np.concatenate([pose, face, lh, rh])
            sequence.append(keypoints)

    cap.release()

    if len(sequence) < 90:
        return None # Not enough frames
    
    sequence = sequence[:90] # Use only the first 90 frames
    sequence = np.array(sequence) # (90, 2172) shape
    return np.array(sequence)

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No hay ningún video en la solicitud'}), 400

    video = request.files['video']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video.save(tmp.name)
        video_path = tmp.name

    keypoints_seq = extract_keypoints_from_video(video_path)
    os.remove(video_path)

    if keypoints_seq is None:
        return jsonify({'error': 'Video muy corto o mal grabado'}), 400

    input_data = np.expand_dims(keypoints_seq, axis = 0)
    prediction = model.predict(input_data)[0]
    class_id = np.argmax(prediction)
    confidence = float(prediction[class_id])
    confidence = round(confidence * 100, 2)

    predicted_label = label_encoder.inverse_transform([class_id])[0]

    return jsonify({'🎯 Predicción': predicted_label, '📔 Confianza': confidence})

if __name__ == '__main__':
    app.run(debug=True)