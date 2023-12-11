from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Load the saved model
saved_model_path = 'flower_recognition_model.h5'
model = load_model(saved_model_path)

img_size = (224, 224)

# Function to preprocess an image for prediction
def preprocess_image(img):
    img = image.load_img(io.BytesIO(img), target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

app = FastAPI()

@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    # Preprocess the uploaded image for prediction
    img_array = preprocess_image(await file.read())

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    flower_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  

    # Return the prediction results
    return JSONResponse(content={"predicted_class": flower_classes[predicted_class],
                                 "probability": float(predictions[0, predicted_class])})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
