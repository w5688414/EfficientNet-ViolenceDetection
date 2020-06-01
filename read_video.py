import numpy as np
import cv2
import sys

video = "/home/eric/data/violence_recognition/HockeyFights/fi46_xvid.avi"

video_capture = cv2.VideoCapture(video)
if not video_capture.isOpened():
    print("Error: Failed to open %s" % video)
    sys.exit(-1)
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
while(True):
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        count += 1

print(video_length, count)
# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()