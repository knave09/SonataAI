import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
from skimage.transform import resize
import io
import tensorflow as tf

app = Flask(__name__)
CORS(app)

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load model only when needed
model = None

def load_model():
    global model
    if model is None:
        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        model = interpreter
    return model

def load_and_preprocess_file(file, target_shape=(210, 210)):
    data = []
    audio_data, sample_rate = librosa.load(file, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        melspectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        melspectrogram = resize(np.expand_dims(melspectrogram, axis=-1), target_shape)
        data.append(melspectrogram)
    return np.array(data)

def model_prediction(X_test):
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], X_test.astype(np.float32))
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    
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
            file_content = io.BytesIO(file.read())
            X_test = load_and_preprocess_file(file_content)
            c_index = model_prediction(X_test)
            predicted_genre = classes[c_index]
            
            return jsonify({'genre': predicted_genre})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)