from flask import Flask, render_template, request, jsonify
import face_recognition
import pickle
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
with open('trained_svm_model.pkl', 'rb') as f:
    clf = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    if 'webcam_image' not in request.files:
        return jsonify({"error": "No image uploaded!"}), 400

    # Get the image from the request
    webcam_image = request.files['webcam_image']

    # Convert the image to a format that face_recognition can work with
    image_bytes = webcam_image.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    # Use face_recognition to find faces and encode them
    face_encodings = face_recognition.face_encodings(image_np)

    if not face_encodings:
        return jsonify({"error": "No face detected!"}), 400

    # Use the trained SVM model to predict the person
    predicted_name = clf.predict([face_encodings[0]])[0]
    
    return jsonify({"message": f"Predicted person: {predicted_name}"}), 200

if __name__ == '__main__':
    app.run(debug=True)