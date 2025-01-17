<div id="header" align="center">
  <h1>
    C-elegans Tracking System
  </h1>
</div>

[![](https://raw.githubusercontent.com/hilariie/C-elegans-tracker/main/videos/thumbnail.png)](https://hilariie.github.io/projects/worm-tracker/videos/c-elegans-overview.mp4)
---
This project aims to detect and track a male C-elegans worm's mating behaviour amongst female worms. It uses Yolov8 by 
Ultralytics to detect the worms, the Segment Anything Model (SAM) to segment the detected worms from background, the DeepSORT algorithm to track the worms, Euclidean pixel distance to differentiate worms (similar in appearance) by mobility, custom algorithms to detect when contact between male and female worms, and records the contact time as well as the male worm's trajectory for scientific research.


# About the Project
This project experiments the combination of different algorithms including the experimentation of a modified version of the DeepSORT and their overall accuracies were compared with the modified DeepSORT coupled with the use of SAM and YOLO having the highest tracking accuracy.

To train the object detection model (YOLOv8), several annotation approaches were used and compared.


### Worm Detection
To access the yolo training process, navigate to the `yolo` directory where you'd immediately meet directories of the different annotation approaches used, each having
the scripts used to train the models, the model weights,
and their performance scores respectively. The two types of annotation approaches used are stated below:

1. Bounding boxes: Worms here are annotated by bounding boxes and are located in the `single_class` directory.
2. Segmentation: Worms here are annotated by segmentation coordinates and are located in the `segmentation` directory. 

### Worm Tracking

DeepSORT tracking algorithm was used to track worms and tracking systems were developed for the two types of annotation techniques. 
#### 1. Bounding Boxes: `worm_tracker.py`
- YOLOv8 was used for worm detection while DeepSORT was used for worm tracking. Results from this script and configuration are found in `results/single_class/normal`
- YOLOv8 was used for worm detetion, SAM for worm segmentations, and DeepSORT for worm tracking. Results from this script and configuration are found in `results/single_class/sam`

#### 2. Segmentation Coordinates: `segmentation_tracker.py`
- YOLOv8 was used for worm detection while DeepSORT was used for worm tracking. Results from this script and configuration are found in `results/segmentation/normal`
- YOLOv8 was used for worm detetion, SAM for worm segmentations, and DeepSORT for worm tracking. Results from this script and configuration are found in `results/segmentation/sam`
###### *Note:*
*There is also an option to choose between the use of our modified DeepSORT algorithm and the default DeepSORT algorithm. The difference in results is found in the video output names. 'default' and 'modified' differentiates between both versions of the algorithm used.*



## Dependencies
`worm_tracker.py` and `segmentation_tracker.py`: To run you need to install the following:
- `python==3.10.11`
- `numpy==1.23.5`
- `ultralytics`
- `TensorFlow`
- `segment-anything-py`

## How to Use
1. Clone the repository using the command: `git clone <repo_url>`
2. Navigate to the project directory: `cd <repo>`
3. Download the videos used for tracking. Unzip and move the videos into the `videos` directory.
4. Modify the `config.yaml` file accordingly to carry out experimentations.
5. Execute the scripts (`worm_tracker.py` or `segmentation_tracker.py`) in accordance to your modifications in the `config.yaml` file.

After execution, the output frames of the video with the tracked worms would be displayed. If you wish to end the script/video, press 'q'.
