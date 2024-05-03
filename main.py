import cv2
from datetime import datetime
from ultralytics import YOLO
from config import *
import time
from mivovo import model_cls
from service.utils import *
model = YOLO('ai_models\yolov8n-face.pt')


def process_image(frame, track):
    start_time = datetime.now()
    results = model.track(frame, persist=True, verbose=False)
    face_id_select =[]
    relationship = ''
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.tolist()
        boxes = results[0].boxes.xyxy.tolist()
        boxes = [list(map(int, box)) for box in boxes]   
        for box, id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            if x2 - x1 > 60 and y2 - y1 > 60:
                face_id_select.append(id)
                face_box = frame[y1:y2, x1:x2]
                if track.get(id) is None:
                    track[id] = {}
                    age, gender = model_cls.predict(face_box)
                    track[id]["age"] = int(age)
                    track[id]["gender"] = gender
                    end_time = datetime.now()
                    track[id]["tracking_time"] = (end_time - start_time).total_seconds() 
                    track[id]["num_tracking"] = 1
                else:
                    end_time = datetime.now()
                    track[id]["tracking_time"] += (end_time - start_time).total_seconds() 
                    track[id]["num_tracking"] += 1
                
                track[id]["box"] = [x1, y1, x2, y2]
            
        relationship = get_relationship(track)
        report_first_info(track, relationship, face_id_select)
    track = report_final_info(track, face_id_select)
    return track, relationship

track = {}
while True:
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f"Error opening video stream for camera {CAM_ID}")
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        track, relationship = process_image(frame, track)
        if IS_SHOW:
            frame = draw(frame, track, relationship)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)  