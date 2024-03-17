# main.py
import cv2
import argparse
from detection import Detection
from tracker import Tracker
from visualisation import Visualisation

def main(video_path: str, model_path: str, output_path: str, device: str, conf_thresh: float, iou_thresh: float, mode: str):

    '''
        main function to handle runnung the file

    '''
    detection = Detection(model_path, device=device, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    visualiser = Visualisation(output_path)
    
    if mode == 'track':
        tracker = Tracker()

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    output_frames = []  # List to store frames for video creation

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detection.run_inference(frame)

        # if mode == 'track':
        #     #tracker.reset()  # Reset trackers for each new frame
        #     for det in detections:
        #         bbox = (det['xmin'], det['ymin'], det['xmax'] - det['xmin'], det['ymax'] - det['ymin'])
        #         tracker.add_tracker(frame, bbox)
        #     tracker.update(frame)
        #     tracks = tracker.get_tracks()
        #     histories = {track['id']: tracker.get_history(track['id']) for track in tracks}  # New line
        #     frame_with_annotations = visualiser.draw_tracks(frame, tracks, histories)  # Updated call
        #     print(tracks)
        #     print(histories)
        #     print(bbox)

        if mode == 'track':
            for det in detections:
                bbox = (det['xmin'], det['ymin'], det['xmax'] - det['xmin'], det['ymax'] - det['ymin'])
                bbox = tuple(map(int, bbox))
                tracker.add_tracker(frame, bbox)

            tracker.update(frame)
            main_tracks = tracker.get_tracks()
            histories = {track['id']: tracker.get_history(track['id']) for track in main_tracks}
            frame_with_annotations = visualiser.draw_tracks(frame.copy(), main_tracks, histories)  # Use a copy
        else:
            frame_with_annotations = visualiser.draw_detections(frame, detections)

        output_frames.append(frame_with_annotations)  # Save frame for video
        visualiser.save_frame(frame_with_annotations, frame_number)
        cv2.imshow('Frame', frame_with_annotations)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    # After processing all frames, save them as a video
    visualiser.save_video(output_frames, output_path, frame_size=(frame.shape[1], frame.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object detection and tracking in video streams.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLOv5 model file.')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save the visualisation frames and output video.')
    parser.add_argument('--device', type=str, default='cpu', help='Computation device (cuda or cpu).')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Confidence threshold for detections.')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--mode', type=str, choices=['detect', 'track'], default='track', help='Operation mode: detect for detection only, track for detection and tracking.')

    args = parser.parse_args()

    main(args.video_path, args.model_path, args.output_path, args.device, args.conf_thresh, args.iou_thresh, args.mode)
