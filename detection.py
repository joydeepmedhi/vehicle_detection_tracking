# detection.py
import torch
import numpy as np
from typing import List, Dict

class Detection:
    """
    A class to handle object detection using a pre-trained YOLOv5 model.

    Attributes:
        model_path (str): The file path to the trained YOLOv5 model.
        device (str): The device to run the model on ('cpu' or 'cuda').
        model (torch.nn.Module): The loaded YOLOv5 model.
    """

    def __init__(self, model_path: str, device: str = 'cuda', conf_thresh: float = 0.5, iou_thresh: float = 0.45):
        """
        Initializes the Detection class with a model path and device.

        Parameters:
            model_path (str): The file path to the trained YOLOv5 model.
            device (str): The device to run the model on ('cpu' or 'cuda').
            conf_thresh (float): Confidence threshold for the model to consider detections.
            iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS).
        """
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True).to(self.device)
        self.model.conf = conf_thresh  # Confidence threshold
        self.model.iou = iou_thresh  # IoU threshold for NMS

    def run_inference(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs inference on a frame and returns detections.

        Parameters:
            frame (np.ndarray): The input frame for object detection.

        Returns:
            List[Dict]: A list of dictionaries, each containing a detection.
        """
        results = self.model(frame)
        detections = results.pandas().xyxy[0]  # Detections in pandas DataFrame
        return detections.to_dict('records')
