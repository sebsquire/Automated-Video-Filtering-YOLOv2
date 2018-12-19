# Automated-Video-Filtering-YOLOv2
Reduces specific manual video inspection task by discarding the majority of meaningless videos.

Title: Automated Video Filter for traffic analysis

Dependencies: Python 3, Darkflow's YOLOv2, OpenCV, NumPy, tqdm, Pandas, glob. 
N.B.: For adequate speed GPU must be set up for use, otherwise set "gpu" in options to 0.0 (in Process.py) to use your CPU and get the kettle on as it'll be a long day.

Problem: A friend of mine mentioned a neighbour of his was receiving so much business traffic to their home run business, significant disruptions were being caused on their usually quiet road (along with the extra noise pollution and lack of available road space). The council seemed unwilling to hear a case against them without concrete evidence so this friend bought and set up a fixed motion detecting camera pointed at the public road. Unfortunately, this resulted in a large number of videos that were not meaningful - people walking their dogs and putting bins out, cars driving to other addresses, etc. This seemed like the perfect use for machine learning image processing algorithms to filter out the majority of meaningless videos and allow for a drastically reduced manual video filtering task.

Solution: Using Darknet's implementation of a pre-trained YOLOv2 model (https://pjreddie.com/darknet/yolov2/) for automated object detection. Possible business traffic (meaningful) and other (not meaningful) videos can then be separated by analysis of bounding box position and dimensions.

Method (techniques explained in Module Descriptions):
 - ~3500 videos/day are produced, presented as two second chunks due to a quirk of the motion detection camera software. Hence, consecutive videos concerning the same object must be merged for easier watchability. This reduces number of videos needed to be inspected by 90%, although overall length of video to be inspected is the same.
 - Merged videos are searched by the YOLOv2 algorithm to find those containing vehicles and frame by frame results are recorded in individual CSVs for each video.
 - CSVs for each video are searched in Python to find those containing vehicle bounding boxes in requisite positions indicating possible business traffic. A list of videos to be manually inspected is presented.

Module Descriptions:
 - RealTimeVidDetect.py: Detects objects in real time for an input video as a demonstration of the YOLOv2 algorithm.
 - MergeVideos.py: Merges consecutive videos for easier inspection (both automated and manual) and easier recording of possible meaningful videos.
 - VidInputDetect.py: Creates a new CSV for each video detailing positions of bounding boxes for each vehicle (car/truck) object detected in each frame.
 - ListOfYVids.py: Searches each CSV for vehicles in requisite locations indicating possible business traffic and creates a list of possible videos.
 - Process.py: Combines previous 3 modules to fully process a batch of input videos and produce a details of videos to be manually inspeced.

Results: Number of videos to be manually inspected decreased by ~98%. This is composed of a ~90% decrease through merging consecutive videos and subsequent ~85% decrease through retaining only videos containing vehicles in positions indicating possible business traffic. Hence, ~50 videos must be manually inspected per day, of which around half contain business traffic.

Why YOLOv2? YOLO is an effective and quick, state-of-the-art, real-time object detection system capable - most applicable given the volume of video to be processed. 

Future Improvements: 
 - Mask-R-CNN yields more detailed results allowing for semantic segmentation of different object instances, leaeding to theoretically perfectly accurate automated video inspection for this application (as vehicle orientation indicated by masks produced can be used to highlight those pulling into the neighbours) but requires significantly more processing time and was deemed inapplicable given the volume of video to be processed per day.
 - Camera position could be moved from ground floor to first which would improve accuracy of business traffic position recognition (although pixel values indicating business traffic would have to be manually modified). 
 - YOLOv2's accuracy is limited severly by the dark in winter months although project extensions making use of CNN's for improving low light images/videos (http://cchen156.web.engr.illinois.edu/SID.html) may greatly decrease this. 
