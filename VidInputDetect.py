import os
import cv2
from darkflow.net.build import TFNet
import time
from tqdm import tqdm               # show progress of video detection


# create new csv for each video named 'objectsmerged-.csv' showing what cars/trucks are detected in each frame
def record_in_csv(opt, vid_csv_path, colours, vid_csv_list, project_location):
    tfnet = TFNet(opt)
    for i in tqdm(vid_csv_list):
        objectsDetected = []
        capture = cv2.VideoCapture(os.path.join(vid_csv_path, i))
        while capture.isOpened():
            stime = time.time()                                   # start frame time
            ret, frame = capture.read()                           # ret = true when vid playing, frame = created frame
            if ret:
                results = tfnet.return_predict(frame)             # create a frame for each prediction
                objectsDetected.append(results)                   # add entry for each object detected
                # next for loop creates frame visual
                for color, result in zip(colours, results):                 # zip makes list of tuples (colors, results)
                    tl = (result['topleft']['x'], result['topleft']['y'])   # top left of frame
                    br = (result['bottomright']['x'], result['bottomright']['y'])  # btm right of frame
                    label = result['label']  # lbl
                    frame = cv2.rectangle(frame, tl, br, color, 7)          # create frame
                    frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2) # put text on frame
                cv2.imshow('frame', frame)                                  # display frame
                print('FPS {:.1f}'.format(1 / (time.time() - stime)))       # showing FPS
                if cv2.waitKey(1) & 0xFF == ord('q'):                       # stop if you press 'q'
                    break
            else:
                capture.release()                                           # stop when video finishes
                cv2.destroyAllWindows()
                break
        # record in csv
        with open(os.path.join(project_location, 'processed_CSVs\\objects{}.csv'
                          .format(i[:-4])), 'w') as f:  # -4 as input=.avi
            f.write('frame,label,confidence,topleft_x,topleft_y,bttmright_x,bttmright_y,class\n')         # titles
        with open(os.path.join(project_location, 'processed_CSVs\\objects{}.csv'
                          .format(i[:-4])), 'a') as f: # -4 as input=.avi
            for obj_in_frame in objectsDetected:
                if len(obj_in_frame) == 0:                              # no objects detected in current frame
                    f.write('-,-,-,-,-,-,-\n')                          # print dashes for all attributes
                else:                                                   # len(obj_in_frame) != 0
                    for obj in range(0, len(obj_in_frame)):             # for every object detected in frame
                        # only record if label == car or label == truck
                        if obj_in_frame[obj]['label'] == 'car' or obj_in_frame[obj]['label'] == 'truck':
                            f.write('{},{},{},{},{},{},{}\n'.format(objectsDetected.index(obj_in_frame) + 1,
                                                                    obj_in_frame[obj]['label'],
                                                                    obj_in_frame[obj]['confidence'],
                                                                    obj_in_frame[obj]['topleft']['x'],
                                                                    obj_in_frame[obj]['topleft']['y'],
                                                                    obj_in_frame[obj]['bottomright']['x'],
                                                                    obj_in_frame[obj]['bottomright']['y']))
                        else: continue
