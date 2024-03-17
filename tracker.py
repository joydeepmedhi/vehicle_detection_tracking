import cv2
from typing import List, Dict, Tuple

class Tracker:
    """
    Handles multiple object trackers using the KCF algorithm, with unique IDs, history
    tracking, and additional features for robustness and debugging.
    """
    def __init__(self, max_history_length: int = 100, failure_threshold: int = 5,
                 iou_threshold: float = 0.3):
        """
        Initializes the Tracker class.

        Args:
            max_history_length (int): Maximum length of track history to store.
            failure_threshold (int): Consecutive update failures before a tracker is considered lost.
            iou_threshold (float): IoU threshold to suppress duplicate trackers.
        """
        self.trackers = []
        self.id_counter = 0
        self.history = {}
        self.tracks = []
        self.max_history_length = max_history_length
        self.failure_threshold = failure_threshold
        self.iou_threshold = iou_threshold

    def add_tracker(self, frame: cv2.Mat, bbox: Tuple[int, int, int, int]):
        """
        Initializes and adds a new tracker for the object defined by bbox.

        Args:
            frame (cv2.Mat): The video frame to initialize the tracker on.
            bbox (Tuple[int, int, int, int]): Bounding box (x, y, width, height).
        """

        # Check for overlap with existing trackers (prevents duplicates)
        if any(self._calculate_iou(bbox, t['bbox']) > self.iou_threshold for t in self.tracks):
            return  # Skip if overlap is too high

        tracker = cv2.TrackerKCF_create()
        success = True
        tracker.init(frame, bbox)
        # print('success: ', success)
        # tracker = cv2.TrackerCSRT_create()
        # success = tracker.init(frame, bbox)
        # print('Initialization success:', success)

        if success:
            self.trackers.append((tracker, self.id_counter, 0))  # Add failure count
            self.history[self.id_counter] = [bbox]
            self.id_counter += 1
            print(f"Tracker {self.id_counter - 1} initialized")

    def update(self, frame: cv2.Mat):
        """
        Updates all trackers with the current frame.

        Args:
            frame (cv2.Mat): The current video frame.
        """
        self.tracks = []
        for i, (tracker, id_, failure_count) in enumerate(self.trackers):
            success, bbox = tracker.update(frame)
            if success:
                self.tracks.append({'id': id_, 'bbox': bbox, 'success': True})
                self.history[id_].append(bbox)
                self.history[id_] = self.history[id_][-self.max_history_length:]  
                self.trackers[i] = (tracker, id_, 0)  # Reset failure count
            else:
                self.tracks.append({'id': id_, 'bbox': self.history[id_][-1], 'success': False})
                self.trackers[i] = (tracker, id_, failure_count + 1)  

                # Remove if failures exceed threshold:
                if self.trackers[i][2] >= self.failure_threshold:
                    print(f"Tracker {id_} removed due to failures")
                    del self.trackers[i]
                    del self.history[id_]

    def get_tracks(self) -> List[Dict]:
        """
        Returns the current tracking information.
        """
        return self.tracks

    def get_history(self, track_id: int) -> List[Tuple[int, int, int, int]]:
        """
        Retrieves the positional history for a specified track ID.
        """
        return self.history.get(track_id, [])

    def reset(self):
        """
        Clears all trackers and resets the class to its initial state.
        """
        self.trackers = []
        self.id_counter = 0
        self.history = {}
        self.tracks = []

    def _calculate_iou(self, box1, box2) -> float:
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (Tuple[int, int, int, int]): Bounding box 1 (x, y, width, height).
            box2 (Tuple[int, int, int, int]): Bounding box 2 (x, y, width, height).

        Returns:
            float: The IoU value between the two boxes.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection coordinates
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        intersection_area = x_overlap * y_overlap

        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0  

        iou = intersection_area / union_area
        return iou

if __name__ == '__main__':
    # Sample usage for testing 
    frame = cv2.imread("data/frame_0002.jpg")  # Replace with your image path
    if frame is None:
        print("Error loading image.")
    else:
        bbox = (50, 150, 80, 60)  # Sample bounding box
        tracker = Tracker()
        tracker.add_tracker(frame, bbox)

        print(tracker.get_tracks())

        # Pretend to update the tracker a few times (you'll need more frames for real usage)
        for _ in range(5):
            tracker.update(frame)
            tracks = tracker.get_tracks()
            print(tracks) 
