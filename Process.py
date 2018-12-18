'''
Process:
        - Merges Videos
        - Detects and records in CSVs details of cars and trucks found in videos presented in processed_CSVs folder
        - Splits CSVs into those containing vehicles in the requisite locations indicating business traffic, and other
        - Provides list of meaningful, merged videos containing possible business traffic
'''
import os
import numpy as np
import pandas as pd
from glob import glob
from MergeVideos import diff_list, merge
from VidInputDetect import record_in_csv
from ListOfYVids import select_csv
import csv

# location of project and darkflow model folders
project_loc = 'C:\\Users\\squir\\PycharmProjects\\keras_projects\\MotionDetect'
model_loc = 'C:\\Users\\squir\\darkflow-master'

'''              USER DECISIONS              '''
user_decisions = []
print('Would you like to merge consecutive videos? (Y/N)')
user_decisions.append(input())
print('Would you like to detect frames in videos and create CSVs containing details? (Y/N)')
user_decisions.append(input())
print('Would you like to create list of business traffic/other? (Y/N)')
user_decisions.append(input())

'''              Merging videos - OPTIONS                    '''
# next two lines = loading the videos.
unmerged_folder = os.path.join(project_loc, 'unmerged')
raw_vids = os.listdir(unmerged_folder)
# destination path of merged videos
merged_folder = os.path.join(project_loc, 'merged')

'''              Detecting cars and trucks in videos, returning CSV for each video containing details - OPTIONS    '''
# change working directory to place of model
os.chdir(model_loc)
# defining options for image rendering
options = {
    "model": "cfg/yolo.cfg",            # which model to use
    "load": "bin/yolov2.weights",       # which preconfigured weights to use
    "threshold": 0.3,                   # must have confidence factor of equal to or greater than to draw bbox
    "gpu": 1.0}                         # 1 = use the gpu to render, 0 = use CPU
# random colours for each object
create_colours = [tuple(255 * np.random.rand(3)) for i in range(100)]

'''               Creating a list of videos containing possible business traffic - OPTIONS              '''
''' N.B. R.E.: min_pix, max_pix:
They define acceptable range of top left and bottom right box corners indicating car/truck pulling into business.
These have been found through manual inspection of csvs produced using OpenCV and the merged videos
and would have to be modified (manually) for other applications/movement of the camera.
 '''
min_pix = {'topleft_x': 897, 'topleft_y': 534, 'bttmright_x': 1020, 'bttmright_y': 613}
max_pix = {'topleft_x': 1020, 'topleft_y': 592, 'bttmright_x': 1093, 'bttmright_y': 637}
# create empty lists for business traffic, other to be appended with digits indicating csv number
bus_trf, other = [], []

if __name__ == '__main__':
    if user_decisions[0] == 'Y' or user_decisions[0] == 'y':
        # create list of times when vids were last modified (i.e. created)
        raw_vids_w_timediff = diff_list(raw_vids, unmerged_folder)
        # merge consecutive videos
        merge(raw_vids_w_timediff, merged_folder, unmerged_folder)
    elif user_decisions[0] == 'N' or user_decisions[0] == 'n':
        pass
    else: raise Exception('Q1: Please input Y or N')

    if user_decisions[1] == 'Y' or user_decisions[1] == 'y':
        # list of merged video names and location
        merged_vids = os.listdir(merged_folder)
        # create new csv for each video named 'objectsmerged-.csv' showing what objects are detected in each frame
        record_in_csv(options, merged_folder, create_colours, merged_vids, project_loc)
    elif user_decisions[1] == 'N' or user_decisions[1] == 'n':
        pass
    else: raise Exception('Q2: Please input Y or N')

    if user_decisions[2] == 'Y' or user_decisions[2] == 'y':
        for record in glob(os.path.join(project_loc, 'processed_CSVs\\*.csv')):
            current_csv = pd.read_csv(record)
            df = pd.DataFrame(current_csv)
            select_csv(record, df, min_pix, max_pix, bus_trf, other)
        # write results to CSV
        with open(os.path.join(project_loc, 'Videos_to_check.csv'), 'w') as f:
            wr = csv.writer(f, delimiter='\n', quoting=csv.QUOTE_ALL)
            wr.writerow(sorted(bus_trf))
        print('Original number of videos to be checked (automated): ', len(os.listdir(unmerged_folder)))
        print('Number of merged videos to be checked (automated): ', len(os.listdir(merged_folder)))
        print('Processed number of videos to check (manually): ', len(bus_trf))
    elif user_decisions[2] == 'N' or user_decisions[2] == 'n':
        pass
    else: raise Exception('Q3: Please input Y or N')
