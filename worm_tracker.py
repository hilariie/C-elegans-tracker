import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import functions as ft
import yaml
import time

t1 = time.time()

with open('config.yaml', 'r') as file:
    yml = yaml.safe_load(file)

test = yml['test']
if test:
    video_name = 'mating7'
else:
    video_name = yml['video_name']
video_path = f'videos/{video_name}.wmv'

cap = cv2.VideoCapture(video_path)
# skip the part where they set up the worms
setup, time_diff = ft.video_setup(video_name)
cap.set(cv2.CAP_PROP_POS_FRAMES, setup)
ret, frame = cap.read()
# set variables to be used in video processing
tracker = ft.Tracker()
track_dist = {}
objects = {}
colors = {}
trajectory = {}
contacts = []
male_id = None
tracker_accuracy = []
font = cv2.FONT_HERSHEY_SIMPLEX
line = cv2.LINE_AA
contact_time_frames = [(28, 37), (499, 526), (591, 611), (633, 668), (850, 1200)]
TP, TN, FP, FN = 0, 0, 0, 0

worm_count = 0
n_init = 10
worm_check = yml['modify']
SAM = yml['worm_tracker']['sam']

yolo_model_dir = f'yolo/single_class/runs/detect/coloured_detection/weights/best.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if SAM:
    output = 'sam'
    sam = sam_model_registry["vit_b"](checkpoint="sam_models/model_b.pth").to(device)
    mask_predictor = SamPredictor(sam)
    contact_colour = (0, 0, 255)
else:
    output = 'normal'
    contact_colour = 0
model = YOLO(yolo_model_dir)
if worm_check:
    final_str = '_modified'
else:
    final_str = '_default'
output_path = f"results/single_class/{output}/{video_name}{final_str}.mp4"

# set output writer
H, W = frame.shape[:2]
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10.5, (W, H))

while ret:
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES) - setup
    time_elapsed = frame_count // 10.5

    # get predictions
    results = model(frame)[0]
    # list to store predicted bbox and scores of each worm
    detections = []
    for result in results.boxes.data.tolist():
        # Get bounding box coordinates, confidence score, and class id for each detected worm
        x1, y1, x2, y2, score, class_id = result
        detections.append([x1, y1, x2, y2, score, class_id])
    # check if worm_check is True, so we do not perform this operation all the time.
    if worm_check:
        # Check if all six worms are being tracked by DeepSORT
        worm_count, n_init, worm_check = ft.worm_checker(len(detections), worm_count, n_init, worm_check)
    if SAM:  # If SAM is True, perform image segmentation using SAM.
        frame = ft.sam_segmentation(detections, frame, mask_predictor)

    tracker.update(frame, detections, n_init)

    male_id = ft.draw_tracks_and_trajectories(tracker, trajectory, frame, colors, objects, frame_count, results.names,
                                              male_id, track_dist)

    if frame_count > 21:
        # Compute euclidean distance between male and female worms
        contact_detect = ft.bbox_contact(trajectory, male_id, objects, frame_count, contacts, frame, contact_colour)
        if test:
            # Calculate and display tracking accuracy
            ft.tracking_accuracy(tracker.tracks, tracker_accuracy, objects)
            cv2.putText(frame, f'MOTA Acc: {round(np.mean(tracker_accuracy), 1)}%', (W - 310, 60), font, 1, contact_colour,
                        2, line)

            # Calculate and display contact detection accuracy
            contact_acc, f1_acc, TP, FP, TN, FN = ft.contact_evaluation(TP, FP, FN, TN, contact_detect,
                                                                        video_name, time_elapsed, time_diff)
            if f1_acc:
                cv2.putText(frame, f'Contact Acc: {round(contact_acc, 1)}%', (W - 330, 100), font, 1, contact_colour, 2,
                            line)
                cv2.putText(frame, f'F1 Score: {round(f1_acc, 1)}%', (W - 330, 140), font, 1, contact_colour, 2, line)

    # Resize and then display frame and write to output video
    display_frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('C-elegan DeepSORT Tracker', display_frame)
    cap_out.write(frame)
    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

print(f"Time Elapsed: {time.time() - t1}")

male_trajectory = trajectory[male_id]
# write male trajectory to text file
male_path = f"results/single_class/{output}/{video_name}{final_str}_trajectory.txt"
male_trajectory = ','.join([str(trajectories) for trajectories in male_trajectory]).replace('),', ')\n')
with open(male_path, "w") as male_out:
    male_out.write(male_trajectory)


# write time of contact to text file
contact_path = f"results/single_class/{output}/{video_name}{final_str}.txt"
contacts = sorted(set(contacts))
contacts = ','.join(contacts).replace(',', '\n')
with open(contact_path, "w") as contacts_out:
    contacts_out.write(contacts)

if test:
    # Confusion matrix for contact detection
    conf_matrix = np.array([[TP, FN], [FP, TN]])

    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Contact', 'No Contact'],
                yticklabels=['Actual Contact', 'Actual No contact'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.savefig(f"results/single_class/{output}/{video_name}{final_str}_confusion_matrix.png")
    plt.show()
