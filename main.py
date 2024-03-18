import numpy as np
from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('models/license_plate_detector.pt')

cap = cv2.VideoCapture('V1.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detections in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detections
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        track_ids = mot_tracker.update(np.asarray(detections_))

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            license_plates_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            license_plates_crop_gray = cv2.cvtColor(license_plates_crop, cv2.COLOR_BGR2GRAY)
            _, license_plates_crop_thresh = cv2.threshold(license_plates_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            license_plates_text, license_plates_text_score = read_license_plate(license_plates_crop_thresh)

            if license_plates_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plates_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plates_text_score}}

write_csv(results, './test.csv')  # Moved outside the loop to ensure it's executed after all frames are processed
