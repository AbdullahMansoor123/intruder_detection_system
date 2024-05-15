import os, sys
import cv2
from exception import CustomException

from ultralytics import YOLO

from config import *
from utils import draw_roi,region_checker, plot_bboxes, video_writer ,drawBannerText



#capture video
video_cap = cv2.VideoCapture(sample_video)

# Call the function to draw ROI on the first frame
ret, frame1 = video_cap.read()
roi = draw_roi(frame1)

try:
    
    while True:
        ret, frame = video_cap.read()
        frame_org = frame.copy()
        if ret is None:
            break
        
        model = YOLO(model_weights)
        person_cls = [0]  # Person only
        results = model.predict(frame, conf=0.8, classes= person_cls)
        frame = plot_bboxes(results, frame, conf_thres=0.3)
        frame = region_checker(frame, results, roi)
        cv2.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[0]+roi[2]), int(roi[1]+roi[3])), green, 2)
        

        # write the video
        vid_write_object = video_writer(video_cap=video_cap)
        vid_write_object.write(frame)

        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_cap.release()
    cv2.destroyAllWindows()

 
except Exception as e:
    raise CustomException(e,sys)