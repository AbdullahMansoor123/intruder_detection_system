import os
import cv2
import numpy as np
from config import * 

def video_writer(video_cap):
    frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    size = (frame_w, frame_h)
    
    output_path = 'output_videos'
    os.makedirs(output_path,exist_ok=True)
    output_video_path = os.path.join(output_path, 'output.mp4')
    output_video_obj = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, size)
    return output_video_obj


def draw_roi(image):
    # Display the image
    cv2.imshow('Select ROI', image)
    
    # Select ROI
    roi = cv2.selectROI('Select ROI', image, fromCenter=False, showCrosshair=True)
    
    # Close the window
    cv2.destroyAllWindows()
    return roi


def region_checker(frame, results, roi):
    """
    :param zone: Zone under observation
    :param row:  information of each prediction
    :param zc: zone color
    :return: draw and change person bbox when enter specific zone
    """
    zpt = [(int(roi[0]), int(roi[1])), 
           (int(roi[0])+int(roi[2]), int(roi[1])), 
           (int(roi[0])+int(roi[2]),int(roi[1]+roi[3])),
           (int(roi[0]),int(roi[1]+roi[3]))]
    for result in results[0]:
        bbox = result.boxes.data[0][:4]

        # row: coordinates of the detected person
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bbox_point = int(x2), int((y1+y2)/2)
        inside_region = cv2.pointPolygonTest(np.array(zpt), bbox_point, False)

        # check if person is inside the zone
        if inside_region > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), red, 1)
            drawBannerText(frame, 'Intruder ALERT!!!', text_color=red)
        else:
            pass
    return frame

def display_text(frame, text, left, top, color):
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    cv2.rectangle(frame, (left, top), (left + text_w, top - text_h), color, -1)
    cv2.putText(frame, text, (left, top), fontFace, fontScale, (255, 255, 255), fontThickness, cv2.LINE_AA)


def plot_bboxes(results, frame, last_bbox):
    """
    :param results: logits from the model
    :param frame: frame of video after point polygon check
    :return:
    """
    
    yolo_class = ['person']
    conf_thres = 0.3
    textSize = cv2.getTextSize(yolo_class[0], fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    
    for result in results[0]:
        bbox = result.boxes.data[0][:4]
        conf = result.boxes.data[0][4]
        label = result.boxes.data[0][5]
        
        if len(bbox) ==0:
            bbox = last_bbox
        else:
            last_bbox = bbox

        if conf >= conf_thres:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # for Free Zone ##
            cv2.rectangle(frame, (x1, y1), (x2, y2), green, 1)
            # display_text(frame, f'{conf}', x1, y1, (255, 255, 255))

    return frame, last_bbox


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

