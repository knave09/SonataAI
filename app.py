import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import librosa
from skimage.transform import resize
import io

app = Flask(__name__)
CORS(app)

# Load the model
model = load_model('trained_model.keras')

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_and_preprocess_file(file, target_shape=(210, 210)):
    data = []
    audio_data, sample_rate = librosa.load(file, sr=None)
    #define chunk and overlap duration
    chunk_duration = 4
    overlap_duration = 2
    
    #Convert duration to sample
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    #Calculate number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    #Iterate over each chunks
    for i in range(num_chunks):
        #Calculate the start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        #Extract the chunk audio
        chunk = audio_data[start:end]
        melspectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        melspectrogram = resize(np.expand_dims(melspectrogram, axis=-1), target_shape)
        #Append data to list
        data.append(melspectrogram)
    return np.array(data)

def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts==max_count]
    return max_elements[0]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Load and preprocess the audio file
            file_content = io.BytesIO(file.read())
            X_test = load_and_preprocess_file(file_content)
            
            # Make prediction
            c_index = model_prediction(X_test)
            predicted_genre = classes[c_index]
            
            return jsonify({'genre': predicted_genre})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)