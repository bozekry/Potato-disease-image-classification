import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the pre-trained Keras model
model = keras.models.load_model('my_image_classification_model.h5')

# Define the class labels
class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'] 

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to 224x224 pixels
    image = image.resize((256,256))
    # Convert the PIL image to a numpy array
    img_array = np.array(image)
    # Reshape the array to (1,256,256, 3) to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image using the same preprocessing function used during training
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Define the Streamlit app

st.title('Image Classification App')
st.write('Upload an image and the app will predict the class label.')
# Allow the user to upload an image
uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Load the image into PIL format
    image = Image.open(uploaded_file)
    # Preprocess the image
    img_array = preprocess_image(image)
    # Make a prediction using the model
    predictions = model.predict(img_array)
    # Get the predicted class label
    predicted_class = class_labels[np.argmax(predictions)]
    confidence=round(predictions.max()*100)
    # Display the image and the predicted class label
    st.image(image, caption=f'Predicted class: {predicted_class} <br> Confidence: {confidence}', use_column_width=True)
st.markdown("### by Mahmoud Said")
