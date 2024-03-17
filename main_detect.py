import cv2
import argparse
from detection import Detection
from visualisation import Visualisation

def main(video_path: str, model_path: str, output_path: str, device: str, conf_thresh: float, iou_thresh: float):
    """
    Processes the video to detect objects and visualises the results.

    Parameters:
        video_path (str): Path to the video file.
        model_path (str): Path to the YOLOv5 model file.
        output_path (str): Directory to save the visualisation frames.
        device (str): The computation device ('cuda' or 'cpu').
        conf_thresh (float): Confidence threshold for detections.
        iou_thresh (float): IoU threshold for Non-Maximum Suppression.
    """
    detection = Detection(model_path, device=device, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    visualisation = Visualisation(output_path)

    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        detections = detection.run_inference(frame)
        
        # Draw detections on the frame
        frame_with_detections = visualisation.draw_detections(frame, detections)
        
        # Optionally, save the frame with visualisations
        visualisation.save_frame(frame_with_detections, frame_number)
        
        # Display the frame
        cv2.imshow('Frame', frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in a video stream and visualise results.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLOv5 model file.')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save the visualisation frames.')
    parser.add_argument('--device', type=str, default='cpu', help='Computation device (cuda or cpu).')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Confidence threshold for detections.')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='IoU threshold for Non-Maximum Suppression.')

    args = parser.parse_args()

    main(args.video_path, args.model_path, args.output_path, args.device, args.conf_thresh, args.iou_thresh)
