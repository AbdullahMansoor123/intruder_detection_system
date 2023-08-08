import os
import glob

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


def motion_detector(source, output_write=False, display=False):
    # video capture object

    video_cap = cv2.VideoCapture(source)
    # video writer object
    frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    size = (frame_w, frame_h)
    frame_area = frame_w * frame_h

    output_path = 'output'
    if not os.path.exists('output'):
        os.mkdir('output')
    else:
        pass

    output_video_path = os.path.join(output_path, source) + '_output.mp4'
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, size)

    # background subtractor
    bg_sub = cv2.createBackgroundSubtractorKNN(history=200)

    ksize = (5, 5)
    min_contour_limit = 0.01
    max_contours = 3
    yellow = (0, 255, 255)
    red = (0, 0, 255)

    # Process video frames
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if frame is None:
            break
        fg_mask = bg_sub.apply(frame)
        fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
        contours, hierarchy = cv2.findContours(fg_mask_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # check for contours
        if len(contours) > 0:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            max_contour_area = cv2.contourArea(sorted_contours[0])
            contour_frac = max_contour_area / frame_area
            if contour_frac > min_contour_limit:
                for i in range(min(max_contours, len(sorted_contours))):
                    xc, yc, wc, hc = cv2.boundingRect(sorted_contours[i])
                    if i == 0:
                        x1 = xc
                        y1 = hc
                        x2 = xc + xc
                        y2 = yc + hc
                    else:
                        x1 = min(x1, xc)
                        y1 = min(y1, yc)
                        x2 = max(x2, xc + wc)
                        y2 = max(y2, yc + hc)
                cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, thickness=2)

                # save video and allow display
                # Display detected image with water mark

                drawBannerText(frame, 'MOVEMENT ALERT!!!', text_color=red)

        # write the video
        if output_write:
            output_video.write(frame)
        if display:
            cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_cap.release()
    cv2.destroyAllWindows()
    output_video.release()


def main():
    source = 'input/intruder_1.mp4'  # change the file path
    motion_detector(source)


if __name__ == "__main__":
    main()
