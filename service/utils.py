import cv2
import numpy as np
from tensorflow.keras.preprocessing import image


def preprocess(current_img):
    target_size = [224, 224]
    if target_size is not None:
        factor_0 = target_size[0] / current_img.shape[0]
        factor_1 = target_size[1] / current_img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (
            int(current_img.shape[1] * factor),
            int(current_img.shape[0] * factor),
        )
        current_img = cv2.resize(current_img, dsize)

        diff_0 = target_size[0] - current_img.shape[0]
        diff_1 = target_size[1] - current_img.shape[1]

        current_img = np.pad(
            current_img,
            (
                (diff_0 // 2, diff_0 - diff_0 // 2),
                (diff_1 // 2, diff_1 - diff_1 // 2),
                (0, 0),
            ),
            "constant",
        )

        # double check: if target image is not still the same size with target.
        if current_img.shape[0:2] != target_size:
            current_img = cv2.resize(current_img, target_size)
        
        img_pixels = image.img_to_array(current_img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]
    return img_pixels