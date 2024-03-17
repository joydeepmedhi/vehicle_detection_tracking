# visualisation.py
import cv2
import os
import numpy as np
from typing import List, Dict, Tuple

class Visualisation:
    """
    A class for visualising detections and tracks on video frames, and optionally saving the frames.
    """

    def __init__(self, output_path: str):
        """
        Initializes the Visualisation class with an output directory path.
        
        Parameters:
            output_path (str): The directory path where output frames are saved.
        """
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def draw_detections(self, frame: np.ndarray, detections: List[Dict], color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
        """
        Draws detections on a frame.

        Parameters:
            frame (np.ndarray): The video frame to draw detections on.
            detections (List[Dict]): The detections to draw. Each detection is a dictionary containing at least 'xmin', 'ymin', 'xmax', 'ymax'.
            color (Tuple[int, int, int], optional): The color to draw the detection rectangles. Defaults to blue.
            thickness (int, optional): The thickness of the detection rectangles. Defaults to 2.

        Returns:
            np.ndarray: The frame with detections drawn.
        """
        for det in detections:
            cv2.rectangle(frame, (int(det['xmin']), int(det['ymin'])), (int(det['xmax']), int(det['ymax'])), color, thickness)
            if 'confidence' in det:
                label = f"{det.get('class', 'Object')} {det['confidence']:.2f}"
                cv2.putText(frame, label, (int(det['xmin']), int(det['ymin'])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        return frame

    def draw_tracks(frame: np.ndarray, tracks: List[Dict], histories: Dict[int, List[Tuple[int, int, int, int]]],
                color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2, history_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Draws tracks, their historical paths, and track IDs on a frame.

        Args:
            frame (np2array): The video frame to draw tracks on.
            tracks (List[Dict]): The tracks to draw, containing 'id', 'bbox', and 'success' status.
            histories (Dict[int, List[Tuple[int, int, int, int]]]): Track history.
            color (Tuple[int, int, int]): Color for current bounding boxes.
            thickness (int): Thickness of bounding boxes.
            history_color (Tuple[int, int, int]): Color for track history.
        """
        print("####", type(tracks))

        print("function tracks", tracks)

        for track in tracks:
            print("type of track", type(track))
            bbox = track['bbox']
            track_id = track['id']

            # Draw bounding box
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, thickness)

            # Draw ID
            cv2.putText(frame, str(track_id), (int(bbox[0]), int(bbox[1] - 5)), 
                        cv2.FONT_HERSHEY_PLAIN, 1, color, thickness)

            # Draw history
            history = histories.get(track_id, [])
            for i in range(1, len(history)):
                cv2.line(frame, (int(history[i - 1][0] + history[i - 1][2] / 2), int(history[i - 1][1] + history[i - 1][3] / 2)),
                         (int(history[i][0] + history[i][2] / 2), int(history[i][1] + history[i][3] / 2)), history_color, thickness)
        return frame

    def save_frame(self, frame: np.ndarray, frame_number: int, filename_prefix: str = "frame") -> None:
        """
        Saves a frame to the output directory.

        Parameters:
            frame (np.ndarray): The frame to save.
            frame_number (int): The frame number, used to generate the filename.
            filename_prefix (str, optional): The prefix for the filename. Defaults to 'frame'.
        """
        frame_path = os.path.join(self.output_path, f"{filename_prefix}_{frame_number:04d}.jpg")

    def save_video(frames: List[np.ndarray], output_path: str, frame_size: Tuple[int, int], fps: int = 24) -> None:
        """
        Saves the processed frames as a video.

        Parameters:
            frames (List[np.ndarray]): The frames to save as a video.
            output_path (str): The directory where the output video is saved.
            frame_size (Tuple[int, int]): The size of the frames.
            fps (int): The frames per second of the output video.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        out = cv2.VideoWriter(f'{output_path}/output_video.mp4', fourcc, fps, frame_size)

        for frame in frames:
            out.write(frame)

        out.release()
        print("Video saved to:", f'{output_path}/output_video.mp4')