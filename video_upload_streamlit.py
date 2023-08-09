import streamlit as st
import cv2 as cv
import tempfile

f = st.file_uploader("Upload file")
tfile = tempfile.NamedTemporaryFile(delete=False)

tfile.write(f.read())
vf = cv.VideoCapture(tfile.name)


# #method #1
stframe = st.empty()
while vf.isOpened():
    ret, frame = vf.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    stframe.image(gray)

#method#2
# video_file = open(tfile.name, 'rb')
# video_bytes = video_file.read()
# st.write(video_bytes)
# if video_bytes is not None:
# st.video(video_bytes)
# else:
#     st.write("No video uploaded")