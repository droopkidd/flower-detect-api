from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved model
saved_model_path = 'flower_recognition_model.h5'
model = load_model(saved_model_path)

img_size = (224, 224)

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        
        temp_path = 'temp_image.jpg'
        file.save(temp_path)

        img_array = preprocess_image(temp_path)

        predictions = model.predict(img_array)

        predicted_class = np.argmax(predictions)

        flower_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

        response = {
            'predicted_class': flower_classes[predicted_class],
            'probability': float(predictions[0, predicted_class])
        }

        return jsonify(response)
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
