import cv2
import numpy as np
# from tensorflow.keras.preprocessing import image
import requests
from config import URL_DETECT, URL_LOG

# def preprocess(current_img):
#     target_size = [224, 224]
#     if target_size is not None:
#         factor_0 = target_size[0] / current_img.shape[0]
#         factor_1 = target_size[1] / current_img.shape[1]
#         factor = min(factor_0, factor_1)

#         dsize = (
#             int(current_img.shape[1] * factor),
#             int(current_img.shape[0] * factor),
#         )
#         current_img = cv2.resize(current_img, dsize)

#         diff_0 = target_size[0] - current_img.shape[0]
#         diff_1 = target_size[1] - current_img.shape[1]

#         current_img = np.pad(
#             current_img,
#             (
#                 (diff_0 // 2, diff_0 - diff_0 // 2),
#                 (diff_1 // 2, diff_1 - diff_1 // 2),
#                 (0, 0),
#             ),
#             "constant",
#         )

#         # double check: if target image is not still the same size with target.
#         if current_img.shape[0:2] != target_size:
#             current_img = cv2.resize(current_img, target_size)
        
#         img_pixels = image.img_to_array(current_img)
#         img_pixels = np.expand_dims(img_pixels, axis=0)
#         img_pixels /= 255  # normalize input in [0, 1]
#     return img_pixels

def draw(image, track, relationship):
    for key, track_info in track.items():
        cv2.rectangle(image, (track_info["box"][0], track_info["box"][1]), (track_info["box"][2], track_info["box"][3]), (255, 0, 0), 2)
        cv2.putText(image, str(track_info["age"]), (track_info["box"][0], track_info["box"][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(image, track_info["gender"], (track_info["box"][0], track_info["box"][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(image, str(track_info["tracking_time"]), (track_info["box"][0], track_info["box"][1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(image, relationship, (track_info["box"][0], track_info["box"][1] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image

def get_relationship(genders):
    relationship = "single"
    if len(set(genders)) > 1:
        relationship = "couple"    
    return relationship

def report_first_info(track, relationship):
    for key in list(track.keys()):
        if track[key]["num_tracking"] == 2:
            try:
                requests.post(
                    URL_DETECT,
                    json={
                        "age": track[key]["age"],
                        "gender": track[key]["gender"],
                        "relationship": relationship,
                    },
                )
            except:
                pass

def report_final_info(track, ids):
    for key in list(track.keys()):
        if key not in ids:
            try:
                requests.post(
                    URL_LOG,
                    json={"tracking_time": track[key]["tracking_time"]},
                )
            except:
                pass
            track.pop(key)
    return track