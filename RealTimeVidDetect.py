'''     YOLOv2 model example which detects where objects exist in each frame of a video in real time
        GPU use must be enabled to be performed in real time
        I ran this on Geforce GTX 1060 at runs at 15FPS (real time), on CPU runs @ 1FPS

        To run this:
            - Ensure model location (model_loc) is set to location of model on your PC
            - Change video name (video_name) to name of your video clip and place it in the same folder as the model '''
import os
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

# model and video location & video name
model_loc = 'C:/Users/squir/darkflow-master'
video_name = 'videofile1.avi'

os.chdir(model_loc)
# define video to be detected and colours used in frames
capture = cv2.VideoCapture(video_name)
colours = [tuple(255 * np.random.rand(3)) for i in range(100)]

# defining options for image rendering
options = {
    "model": "cfg/yolo.cfg",            # which model to use
    "load": "bin/yolov2.weights",       # which preconfigured weights to use
    "threshold": 0.3,                   # must have confidence factor of equal to or greater than to draw bbox
    "gpu": 1.0}                         # 1 = use the gpu to render, 0 = use CPU
tfnet = TFNet(options)

while capture.isOpened():                                       # while video is open
    stime = time.time()                                         # start frame time
    ret, frame = capture.read()                                 # ret = true when vid playing, frame = created frame
    if ret:
        results = tfnet.return_predict(frame)                   # create a frame for each prediction
        for color, result in zip(colours, results):              # zip makes list of tuples (colors, results)
            tl = (result['topleft']['x'], result['topleft']['y'])           # top left of frame
            br = (result['bottomright']['x'], result['bottomright']['y'])   # btm right of frame
            label = result['label']                                         # lbl
            frame = cv2.rectangle(frame, tl, br, color, 7)                  # create frame
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)    # label on frame
        cv2.imshow('frame', frame)                                          # display frame
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))               # showing FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):                               # stop if you press 'q'
            break
    else:
        capture.release()                                                   # else stop when video finishes
        cv2.destroyAllWindows()
        break
