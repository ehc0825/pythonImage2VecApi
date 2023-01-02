import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


class ImageParser:
    def imgToModelInput(img):
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        modelInput = preprocess_input(x)
        return modelInput
