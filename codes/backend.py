from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('D:\study\codes\engineering_codes\projects\written_digit_recognition_using_c\codes\model.h5')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    
    # Invert the image (white background, black digit)
    image = Image.eval(image, lambda x: 255 - x)  # Inverts the grayscale image
    
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Add batch and channel dimensions
    # polting the image
    plt.imshow(image_array[0, :, :, 0], cmap='gray')
    plt.show()
    return image_array


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Decode the base64 image data
    image_data = base64.b64decode(data['image'])
    try:
        processed_image = preprocess_image(image_data)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400

    try:
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        print("Raw prediction:", prediction)  
        print(f'Predicted digit: {predicted_digit}')
        return jsonify({'digit': int(predicted_digit)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
