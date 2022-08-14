#from fastai.vision import open_image, load_learner, image, torch
from fastai.vision.all import *
from fastai.data.external import *
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
import PIL.Image
import requests
from io import BytesIO
import tempfile
import wget

# App title
st.title("Pegasus or Unicorn?")


def predict(img, display_img):

    # Display the test image
    st.image(display_img, use_column_width=True)

    # Temporarily displays a message while executing
    with st.spinner('Wait for it...'):
        time.sleep(3)

    # Load model and make prediction
    model = load_learner('model/my_model.pkl')
    pred = model.predict(img)
    pred_class = pred[0]
    prob = pred[2]
    pred_prob = round(torch.max(prob).item()*100)

    # Display the prediction
    if str(pred_class) == 'unicorn':
        st.success("This is a unicorn with the probability of " +
                   str(pred_prob) + '%.')
    else:
        st.success("This is a pegasus with the probability of " +
                   str(pred_prob) + '%.')


# Image source selection
option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose a test image':

    # Test image selection
    test_images = os.listdir('test/')
    test_image = st.selectbox(
        'Please select a test image:', test_images)

    # Read the image
    file_path = 'test/' + test_image
    img = PILImage.create(file_path)
    # Get the image to display
    display_img = mpimg.imread(file_path)

    # Predict and display the image
    predict(img, display_img)

else:
    url = st.text_input("Please input a url:")

    if url != "":
        try:
            # Read image from the url
            response = requests.get(url)
            pil_img = PIL.Image.open(BytesIO(response.content))
            display_img = np.asarray(pil_img)  # Image to display

            #st.image(display_img, use_column_width=True)

            # Transform the image to feed into the model
            #img = pil_img.convert('RGB')
            #img = image.pil2tensor(img, np.float32).div_(255)
            #img = image.Image(img)
            #img = PILImage.create(pil_img)

            # Grab some random images from the internet, and see what our model thinks it is
            images = [url]

            for image_url in images:
                image_path = tempfile.mktemp()
                wget.download(image_url, out=image_path)
                img = PILImage.create(image_path)
                st.image(display_img, use_column_width=True)
                # Predict and display the image
                #predict(img, display_img)

        except:
            st.text("Invalid url!")
