import os
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Activation
from service.base_model import base_model
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ----------------------------------------

# pylint: disable=too-few-public-methods
class ApparentAgeClient():
    """
    Age model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Age"

    def predict(self, img: np.ndarray) -> np.float64:
        age_predictions = self.model.predict(img, verbose=0)[0, :]
        return find_apparent_age(age_predictions)


def load_model():

    model = base_model()

    # --------------------------

    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    age_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights


    age_model.load_weights("ai_models/age_model_weights.h5")

    return age_model

    # --------------------------


def find_apparent_age(age_predictions: np.ndarray) -> np.float64:
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (?)
    Returns:
        apparent_age (float)
    """
    output_indexes = np.array(list(range(0, 101)))
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age
age_model = ApparentAgeClient()