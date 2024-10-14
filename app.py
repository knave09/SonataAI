from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Configure CORS with specific origins
ALLOWED_ORIGINS = ['https://renaissance0ne.github.io', 'http://localhost:3000']
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# Load the trained model
model = load_model('trained_model.keras')

# Define the genres (make sure this matches your model's output)
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(file):
    # Load the audio file
    y, sr = librosa.load(file, duration=30)
    
    # Extract mel-spectrogram features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Ensure the feature shape matches what your model expects
    if mel_spec_db.shape[1] < 1292:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 1292 - mel_spec_db.shape[1])), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :1292]
    
    return mel_spec_db

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    if file:
        # Save the file temporarily
        temp_path = 'temp_audio.mp3'
        file.save(temp_path)
        
        try:
            # Extract features
            features = extract_features(temp_path)
            
            # Make prediction
            prediction = model.predict(np.expand_dims(features, axis=0))
            
            # Get the predicted genre
            predicted_genre_index = np.argmax(prediction[0])
            predicted_genre = genres[predicted_genre_index]
            
            return jsonify({'genre': predicted_genre})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return jsonify({'error': 'Something went wrong'}), 500

if __name__ == '__main__':
    app.run(debug=False)