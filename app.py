import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

model=load_model('my_cnn_model.h5')

def process_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (170, 170))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
st.title("Skin Cancer Classfication :cancer:")
st.write('Upload an image and model predict it is cancer or not')

file=st.file_uploader('Upload an image',type=['jpg','jpeg','png'])


if file is not None:
    img=Image.open(file)
    st.image(img,caption='Uploaded image')
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)

    class_names=['Non-Cancer','Cancer']

    st.write(class_names[predicted_class])

