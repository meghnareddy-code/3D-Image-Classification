import streamlit as st
from img_classification import classification
import keras
from PIL import Image
import numpy as np
from design import *
st.markdown(html_temp, unsafe_allow_html=True)
st.markdown('<p class="text"> CT scans Example </p>',unsafe_allow_html=True)
st.markdown('<p class = "text"> Upload a CT scan Image for image classification as covid or non-covid </p>',unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CT scan ...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
  image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption='Uploaded CT scan.', use_column_width=True)
  st.write("")
  st.markdown('<p class="text"> Classifying... </p>',unsafe_allow_html=True)
  labels = classification(image, 'vgg_pretrained.h5')

  for index, probability in enumerate(labels):
    if probability[1] > 0.5:
      st.markdown('<p class="big-font text">The CT scan shows that the person has <span class="danger"> covid!! </span> </p>', unsafe_allow_html=True)
    else:
      st.markdown('<p class="big-font text">The CT scan shows that the person has <span class = "safe"> no covid  </span></p>', unsafe_allow_html=True)
