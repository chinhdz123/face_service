import cv2
from datetime import datetime
from ultralytics import YOLO
from service.utils import preprocess
from service.face_emotion import face_emotion_model
from service.gender import gender_model
from service.age import age_model
import requests
from config import *
model = YOLO('ai_models\yolov8n-face.pt')

track = {}

while True:
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f"Error opening video stream for camera {CAM_ID}")
    track = {}
    emotion_true = 0
    total = 0
    age_true = 0
    gender_true = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        start_time = datetime.now()
        results = model.track(frame, persist=True, verbose=False)
        if results[0].boxes.id is None:
            ids = []
        else:
            ids = results[0].boxes.id.tolist()
            boxes = results[0].boxes.xyxy.tolist()
            boxes = [list(map(int, box)) for box in boxes]
            genders = []
            for box, id in zip(boxes, ids):
                x, y, x1, y1 = box
                w, h = x1-x, y1-y
                if w > 60 and h > 60:
                    face_box = frame[y:y1, x:x1]
                    face_box = preprocess(face_box)
                    emotion = face_emotion_model.predict(face_box)
                    gender = gender_model.predict(face_box)
                    age = age_model.predict(face_box)
                    end_time = datetime.now() 
                    genders.append(gender)
                    if track.get(id) is None:
                        track[id] = {}
                        track[id]["tracking_time"] = (end_time-start_time).total_seconds() 
                        track[id]["age"] = age
                        track[id]["emotion"] = emotion
                        track[id]["gender"] = gender
                        track[id]["num_tracking"] = 1
                        #api detect
                    else:
                        track[id]["tracking_time"] += (end_time-start_time).total_seconds()
                        track[id]["age"] = age
                        track[id]["emotion"] = emotion
                        track[id]["gender"] = gender
                        track[id]["num_tracking"] += 1
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                

            if len(set(genders)) > 1:
                relationship = "couple"
            else:
                relationship = "single"

            for key in list(track.keys()):
                if track[id]["num_tracking"] == 2 :
                    try:
                        requests.post(URL_DETECT, 
                                    json={"age": track[key]["age"], 
                                    "emotion": track[key]["emotion"], 
                                    "gender": track[key]["gender"],
                                    "relationship": relationship})
                    except:
                        pass
                    
                cv2.putText(frame, str(track[key]['age']), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.putText(frame, track[key]['emotion'], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.putText(frame, track[key]["gender"], (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.putText(frame, str(track[key]['tracking_time']), (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.putText(frame, relationship, (x, y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        for key in list(track.keys()):
            if key not in ids:
                #api log
                try:
                    requests.post(URL_LOG, 
                                json={"tracking_time": track[key]["tracking_time"]})
                except:
                    pass
                track.pop(key)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        