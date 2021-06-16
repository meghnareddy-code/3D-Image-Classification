import tensorflow
import keras
import numpy as np
from PIL import Image
import cv2


def classification(img, weights_file):
    # Load the model
    model = tensorflow.keras.models.load_model(weights_file)

    #turn the image into a numpy array
    image_array = np.array(img)
    image_array  = cv2.resize(image_array,(250,250)) # resize as per model

    #Preprocessing
    x = img_to_array(image_array)  # Numpy array with shape (250, 250, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 250, 250, 3)
    # Rescale by 1/255
    x /= 255

    # Make prediction
    successive_feature_maps = model.predict(x)

    return successive_feature_maps
