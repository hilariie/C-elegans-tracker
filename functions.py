import numpy as np
import cv2
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection


class Tracker:
    """
    A class for tracking objects using the DeepSORT algorithm.

    Attributes:
        tracker (DeepSortTracker): DeepSORT tracker instance for object tracking.
        encoder (gdet.BoxEncoder): Box encoder instance for feature extraction.
        tracks (list): List of active tracks.

    Methods:
        __init__(): Initializes the tracker with DeepSORT parameters.
        update(frame, detections, n_init): Updates the tracker using new frame and detections.
        update_tracks(): Updates the list of active tracks.
    """
    
    def __init__(self):
        """
        Initializes the Tracker with DeepSORT parameters.
        """
        max_cosine_distance = 0.4
        nn_budget = None
        encoder_model_filename = 'mars-small128.pb'
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric, max_age=10000, max_iou_distance=0.9)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections, n_init=10):
        """
        Updates the tracker with new frame and detections.

        Parameters
        ----------
        frame : numpy.ndarray
            Input video frame.
        detections : list.
            List of detected objects' information ([bbox1, .., bbox4, score, class_id]).
        n_init : int
            Number of consecutive detections before track confirmation.

        Returns
        -------
            None
        """
        bboxes = np.asarray([d[:-2] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-2] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            class_id = detections[bbox_id][-1]
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id], class_id))

        self.tracker.predict()
        self.tracker.update(dets, n_init=n_init)
        self.update_tracks()

    def update_tracks(self):
        """ Updates the list of active tracks."""
        tracks = []
        for x, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            class_id = track.class_id
            tracks.append(Track(id, bbox, class_id))

        self.tracks = tracks


class Track:

    def __init__(self, id, bbox, class_id):
        self.track_id = id
        self.bbox = bbox
        self.class_id = class_id


def worm_checker(dets, worm_count, n_init, worm_check):
    """
    Checks if all 6 worms have been detected consecutively, then increase the n_init parameter

    Parameters
    ----------
    dets : int
        number of detected worms
    worm_count : int
        number of times all 6 worms have been detected consecutively
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    worm_check : boolean
        True if all worms have been detected consecutively for n_init number of times, otherwise False

    Returns
    -------
    worm_count : int
        an increment or reinitialization of worm_count
    n_init : int
        an increment or same value of n_init
    worm_check : boolean
        True if all worms have been detected consecutively for n_init number of times, otherwise False
    """
    if dets == 6:
        worm_count += 1
    else:
        worm_count = 0
    if worm_count == n_init + 1:
        n_init = 400
        worm_check = False
    return worm_count, n_init, worm_check


def bbox_contact(trajectory, male_id, objects, frame_count, contacts, frame, contact_colour):
    """
    Detects and highlights contacts between a male worm and female worm based on their coordinates
    Parameters
    ----------
    trajectory : dict
        A dictionary containing worm IDs as keys and their corresponding trajectories as values.
    male_id : int
        The track ID of the male worm.
    objects : dict
        A dictionary mapping worm IDs to their gender (male or female worms).
    frame_count : int
        The current frame count in the video.
    contacts : list
        A list to store the timestamps of detected contacts.
    frame : numpy.ndarray
        The video frame in which the contacts will be highlighted.
    contact_colour : tuple
        A tuple representing the colour (B, G, R values) for highlighting contacts.

    Returns
    -------
    bool
        True if contact is detected otherwise False.

    Notes
    -----
    - Calculates the Euclidean distance between the male worm's and female worms' current positions.
    - Detects a contact if the distance is less than 82.
    - Adds the contact's timestamp to the 'contacts' list in the format "MM:SS".
    - Draws rectangles and labels around the male and female worms to highlight the contact area.
    """
    # Get the coordinates of the male worm's latest position
    male_centers = (trajectory[male_id][-1][0], trajectory[male_id][-1][1])

    # Iterate through each worm's trajectory
    for ids in trajectory.keys():
        # Check if the worm is a female worm
        if objects[ids] == 'female worm':
            # Get the coordinates of the female worm's latest position
            female_centers = (trajectory[ids][-1][0], trajectory[ids][-1][1])

            # Calculate Euclidean distance between male and female worm positions
            dx = abs(male_centers[0] - female_centers[0]) ** 2
            dy = abs(male_centers[1] - female_centers[1]) ** 2
            euc_dist = (dx + dy) ** 0.5

            # Check if a contact is detected based on distance threshold
            if euc_dist < 82:
                # Calculate time elapsed in video
                time_elapsed = frame_count / 10.5
                minutes = int(time_elapsed / 60)
                seconds = int(time_elapsed % 60)
                contacts.append(f"{minutes:02d}:{seconds:02d}")

                # Define bounding box coordinates for contact visualization
                max_x = int(max(male_centers[0], female_centers[0]))
                min_x = int(min(male_centers[0], female_centers[0]))
                max_y = int(max(male_centers[1], female_centers[1]))
                min_y = int(min(male_centers[1], female_centers[1]))

                # Draw rectangle around contact area and add label
                cv2.rectangle(frame, (min_x - 80, min_y - 80), (max_x + 80, max_y + 80), contact_colour, 2)
                cv2.putText(frame, "CONTACT", (min_x - 80, min_y - 83), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            contact_colour, 1, cv2.LINE_AA)
                return True
    return False


def segment_contact_detection(coordinate_list):
    """
    Detects contacts between worms based on their segmentation coordinates.

    Parameters
    ----------
    coordinate_list : list
        A list containing segmentation coordinates of worms. Each worm's segmentation coordinate is a list of (x, y)
        tuples.

    Returns
    -------
    minx : list
        List of minimum x-coordinates for contact bounding boxes.
    miny : list
        List of minimum y-coordinates for contact bounding boxes.
    maxx : list
        List of maximum x-coordinates for contact bounding boxes.
    maxy : list
        List of maximum y-coordinates for contact bounding boxes.
    """
    contact_worm = []
    # Iterate through each pair of segmentation coordinates
    for i in range(len(coordinate_list)):
        for j in range(i + 1, len(coordinate_list)):
            stop = False

            # Compare each coordinate point in the two worm segmentations
            for coord1 in coordinate_list[i]:
                x1, y1 = coord1

                for coord2 in coordinate_list[j]:
                    x2, y2 = coord2

                    # Calculate euclidean distance between all coordinates of both segmentations/worms
                    dx = (x1 - x2) ** 2
                    dy = (y1 - y2) ** 2
                    euc = round((dx + dy) ** 0.5, 5)

                    # If the Euclidean distance is below the threshold, consider it a contact
                    if euc < 4:
                        contact_worm.append(i)
                        contact_worm.append(j)
                        stop = True
                        break
                if stop:
                    break

    minx, miny, maxx, maxy = [], [], [], []

    # Calculate bounding box coordinates for each detected contact worm
    for i in set(contact_worm):
        worm_coordinates = np.transpose(coordinate_list[i])
        x_coordinates, y_coordinates = worm_coordinates[0], worm_coordinates[1]

        min_x = min(x_coordinates)
        min_y = min(y_coordinates)
        max_x = max(x_coordinates)
        max_y = max(y_coordinates)

        minx.append(min_x)
        miny.append(min_y)
        maxx.append(max_x)
        maxy.append(max_y)

    return minx, miny, maxx, maxy


def video_setup(vid_name):
    """
    Calculates the setup duration in frames for each video. The functions identifies videos using the last character of
    its name. Therefore, video names must follow this format: mating1, mating2, mating3, ... mating7.

    Parameters
    ----------
    vid_name : str
        The name of the video.

    Returns
    -------
    setup : int
        The calculated setup duration in frames.
    time_diff: int
        The time in seconds skipped at the beginning of the video.
    """

    vid_dict = {'mating1': 11, 'mating2': 10, 'mating3': 88, 'mating4': 35, 'mating5': 51, 'mating6': 18, 'mating7': 26}

    # Get the setup duration factor based on the last character of the video name
    time_diff = vid_dict[vid_name]

    # Calculate the setup duration in frames using the factor and the frame rate (10.5 fps)
    setup = round(time_diff * 10.5)

    return setup, time_diff


def contact_evaluation(TP, FP, FN, TN, contact_detect, video_name, time_elapsed, time_diff):
    """
    Calculates the contact detection accuracy of the system.

    Parameters
    ----------
    TP : int
        True Positive
    FP: int
        False Positive
    FN: int
        False Negative
    TN: int
        True Negative
    contact_detect: bool
        True if contact is detected else False.
    video_name: str
        The title of the video the system is being evaluated on.
    time_elapsed: float
        Current time elapsed in the video in seconds.
    time_diff: int
        The time in seconds skipped at the beginning of the video.

    Returns
    -------
    contact_acc: float
        The contact detection accuracy.
    f1_acc: float
        The f1 score of the contact detection accuracy.
    TP: int
        The new true positive value.
    FP: int
        The new false positive value.
    TN: int
        The new true negative value.
    FN: int
        The new false negative value.
    """

    ground_truth = {'mating1': [(78, 1200)],
                    'mating2': [],
                    'mating3': [],
                    'mating4': [],
                    'mating5': [],
                    'mating6': [(689, 728), (763, 815), (876, 1200)],
                    'mating7': [(28, 37), (499, 526), (533, 545), (591, 611), (633, 668), (850, 1200)]}
    # Calculate and display contact detection accuracy
    for time_frames in ground_truth[video_name]:
        if time_elapsed <= time_frames[1] - time_diff:
            # True positive
            if (contact_detect is True) and (time_elapsed >= time_frames[0] - time_diff):
                TP += 1
            # False positive
            elif (contact_detect is True) and (time_elapsed <= (time_frames[0] - time_diff)):
                FP += 1
            # True negative
            elif (contact_detect is False) and (time_elapsed <= (time_frames[0] - time_diff)):
                TN += 1
            elif (contact_detect is False) and (time_elapsed >= time_frames[0] - time_diff):
                FN += 1
            break

    try:
        # Calculate accuracy
        contact_acc = ((TP + TN) / (TP + TN + FP + FN)) * 100
        # Calculate F1 score
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_acc = 200 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_acc = None
        contact_acc = None

    return contact_acc, f1_acc, TP, FP, TN, FN


def sam_segmentation(detections, frame, mask_predictor):
    """
    Perform image segmentation on bounding boxes using the SAM model

    Parameters
    ----------
    detections : list
        List of detected objects' information ([bbox1, .., bbox4, score, class_id]).
    frame : numpy.ndarray
        Frame image to perform segmentation on.
    mask_predictor : SamPredictor
        SAM model predictor for segmentation.

    Returns
    -------
    frame : numpy.ndarray
        Frame image with segmentation applied.
    """
    # Prepare the bounding boxes for segmentation
    segment_bbox = np.array(detections)

    # Set the image for mask predictor
    mask_predictor.set_image(frame)

    mask_list = []
    # Iterate over bounding boxes and predict masks
    for xyxy in segment_bbox:
        new_mask, _, _ = mask_predictor.predict(box=xyxy[:4],
                                                multimask_output=True)
        mask_list.append(new_mask)

    # Combine the predicted masks
    combined_masks = np.logical_or.reduce([masks for masks in mask_list])

    # Convert masks to frame format
    new_frame = np.moveaxis(combined_masks, 0, -1)
    new_frame = np.where(new_frame, 255, 0).astype(np.uint8)

    # Update the frame with segmented regions
    frame = new_frame.copy()
    return frame


def draw_tracks_and_trajectories(tracker, trajectory, frame, colours, objects, frame_count, results_dict, male_id,
                                 track_dist):
    """
    Draw tracks, trajectories, and identify the male worm.

    Parameters
    ----------
    tracker : Tracker
        Tracker instance used for object tracking.
    trajectory : dict
        Dictionary that stores the trajectories of different tracks.
    frame : numpy.ndarray
        Frame image to draw tracks and trajectories on.
    colours : dict
        Dictionary to store track colors.
    objects : dict
        Dictionary to store object class information for tracks.
    frame_count : int
        Current frame count.
    results_dict : dict
        Dictionary mapping class IDs to object class names.
    male_id : int
        Track ID of the identified male worm.
    track_dist : dict
        Dictionary to store track distances.

    Returns
    -------
    male_id : int
        Updated track ID of the identified (if identified) male worm.
    """

    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        track_id = track.track_id
        class_id = track.class_id

        if track_id not in trajectory:
            trajectory[track_id] = [(x_center, y_center)]
        if track_id not in colours:
            colours[track_id] = (250, 180, 180)
        if track_id not in objects:
            objects[track_id] = results_dict[int(class_id)]

        trajectory[track_id].append((x_center, y_center))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colours[track_id], 2)
        cv2.circle(frame, (x_center, y_center), 3, colours[track_id], -1)
        cv2.putText(frame, objects[track_id], (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colours[track_id], 1,
                    cv2.LINE_AA)

        if frame_count == 20:
            # calculate the distance between current location and initial location of worms
            x_center0, y_center0 = trajectory[track_id][0]
            dx = abs(x_center - x_center0) ** 2
            dy = abs(y_center - y_center0) ** 2
            eud = (dx + dy) ** 0.5
            track_dist[track_id] = eud
        # male worms are usually more mobile than female worms therefore,
        # in the next frame, we check the list of distances and get the worm that has moved the most (i.e, fastest worm)
        elif frame_count == 21:
            # Get track ID that has moved the most within the first 21 frames
            max_id = max(track_dist, key=lambda k: track_dist[k])
            # Identify the fastest worm as the male worm, others as female worms
            if track_id == max_id:
                objects[track_id] = 'male worm'
                colours[track_id] = (220, 80, 250)
                male_id = track_id
            else:
                objects[track_id] = 'female worm'

        # Show the travel path of the male worm by drawing lines connecting its location points
        if objects[track_id] == 'male worm':
            for i in range(1, len(trajectory[track_id])):
                prev_center = trajectory[track_id][i - 1]
                next_center = trajectory[track_id][i]
                cv2.line(frame, (int(prev_center[0]), int(prev_center[1])),
                         (int(next_center[0]), int(next_center[1])), colours[track_id], 2)
    return male_id


def tracking_accuracy(tracker, tracker_accuracy, objects):
    """
    Calculates the system's tracking accuracy

    Parameters
    ----------
    tracker : Tracker
        Tracker instance used for object tracking.
    tracker_accuracy : list
        List to hold tracking accuracy scores.
    objects : dict
        Dictionary to store information for tracks.

    """
    false_negative = 6 - len(tracker)
    missed_male_worm = 0.25  # Increase false negative on male worms by 25%.
    false_positive = 0

    for track in tracker:
        track_id = track.track_id
        if objects[track_id] == 'male worm':
            missed_male_worm = 0
        elif objects[track_id] != 'female worm':
            false_positive += 1

    # Calculate for MOTA
    track_acc = (false_negative + false_positive + missed_male_worm) / 6
    track_acc = round((1 - track_acc) * 100)
    # append MOTA score to list so we can get the average of the MOTA score.
    tracker_accuracy.append(track_acc)
    
    
