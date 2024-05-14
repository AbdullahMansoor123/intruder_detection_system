import glob
import os
import random

from ultralytics import YOLO
import cv2
import numpy as np





def drawBannerText(frame, text, banner_height_percent=0.08, font_scale=0.8, text_color=(0, 255, 0),
                   font_thickness=2):
    # Draw a black filled banner across the top of the image frame.
    # percent: set the banner height as a percentage of the frame height.
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)

    # Draw text on banner.
    left_offset = 20
    location = (left_offset, int(10 + (banner_height_percent * frame.shape[0]) / 2))
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                font_thickness, cv2.LINE_AA)

def display_text(frame, text, left, top, color):
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    cv2.rectangle(frame, (left, top), (left + text_w, top - text_h), color, -1)
    cv2.putText(frame, text, (left, top), fontFace, fontScale, (255, 255, 255), fontThickness, cv2.LINE_AA)


def plot_bboxes(results, frame, conf_thres):
    """
    :param results: logits from the model
    :param frame: frame of video after point polygon check
    :return:
    """
    
    yolo_class = ['person']
    textSize = cv2.getTextSize(yolo_class[0], fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]

    for result in results[0]:
        bbox = result.boxes.data[0][:4]
        conf = result.boxes.data[0][4]
        label = result.boxes.data[0][5]

        if conf >= conf_thres:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # for Free Zone ##
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            # display_text(frame, f'{conf}', x1, y1, (255, 255, 255))

    return frame




fontFace = 2
fontScale = 0.7
fontThickness = 1

video_cap = cv2.VideoCapture('sample_videos//intruder.mp4')

# video writer object
frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

size = (frame_w, frame_h)
frame_area = frame_w * frame_h

# output_path = 'output_videos'
# os.makedirs(output_path,exist_ok=True)
# output_video_path = os.path.join(output_path, 'output.mp4')
# output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, size)



ksize = (5, 5)
min_contour_limit = 0.01
max_contours = 3

yellow = (0, 255, 255)
red = (0, 0, 255)
while True:
    ret, frame = video_cap.read()
    frame_org = frame.copy()
    if ret is None:
        break
    
    model = YOLO("artifacts/yolov8n.pt")
    person_cls = [0]  # Person only
    results = model.predict(frame, conf=0.8, classes= person_cls)
    frame = plot_bboxes(results, frame, conf_thres=0.4)
            
    # drawBannerText(frame, 'MOVEMENT ALERT!!!', text_color=red)

    # write the video
    # output_video.write(frame)

    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



video_cap.release()
cv2.destroyAllWindows()



