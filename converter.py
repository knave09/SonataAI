import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('trained_model.keras')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)