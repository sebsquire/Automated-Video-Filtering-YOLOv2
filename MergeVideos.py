'''     Merge videos for easier manual inspection when required
        logic = if video starts within 2.5 secs of previous video - merge            '''
import cv2
import os
from tqdm import tqdm


# create list of times when vids were last modified (aka created)
def diff_list(vid_files, unmerged_vids):
    mod_times = []
    for v in vid_files:
        mod_times.append(os.path.getmtime(os.path.join(unmerged_vids, v)))
    # find difference between modified time for current and last video
    diff = [t - s for s, t in zip(mod_times, mod_times[1:])]
    diff.append(0)                          # need to add extra value so doesn't cut final video off early
    videofiles = zip(vid_files, diff)      # zip for access when merging
    return videofiles


# merge consecutive videos
def merge(vid_files, dest_path, unmerged_vids):
    # for naming merged videos
    merged_index = 0
    out = cv2.VideoWriter(os.path.join(dest_path, 'merged{}.avi'.format(merged_index)),
                          cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                          15, (1920, 1080), 1)
    for vid, dif in tqdm(vid_files):                                       # for each video in folder
        capture = cv2.VideoCapture(os.path.join(unmerged_vids, vid))
        # combine 2 second videos
        while capture.isOpened():
            ret, frame = capture.read()
            out.write(frame)
            if not ret:     # end of 2 sec clip
                break
        # if videos are not consecutive, finish current merging and end merging and create new merge video
        if dif >= 2.5:
            merged_index += 1               # for naming
            out.release()
            # video resolution: 1920x1080 px
            out = cv2.VideoWriter(os.path.join(dest_path, 'merged{}.avi'.format(merged_index)),
                                  cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                                  15, (1920, 1080), 1)
            continue
