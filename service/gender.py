
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Activation
from service.base_model import base_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------------

# Labels for the genders that can be detected by the model.
labels = ["Woman", "Man"]

# pylint: disable=too-few-public-methods
class GenderClient():
    """
    Gender model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Gender"

    def predict(self, current_img: np.ndarray) -> np.ndarray:
        
        rs = self.model.predict(current_img, verbose=0)[0, :]
        rs = np.argmax(rs)
        return labels[rs]



def load_model():

    model = base_model()

    # --------------------------

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    gender_model = Model(inputs=model.input, outputs=base_model_output)


    gender_model.load_weights("ai_models\gender_model_weights.h5")

    return gender_model

gender_model = GenderClient()  