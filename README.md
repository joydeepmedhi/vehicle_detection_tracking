# Object Detection and Tracking Project

This project is designed to perform object detection and tracking on video streams using YOLOv5, OpenCV, and PyTorch. It integrates detection, tracking, and visualization functionalities to process video files and output the results with detected objects and their tracks visualized.

## Prerequisites

Before starting, ensure you have either Anaconda or Miniconda installed on your system. These tools allow you to manage environments and dependencies efficiently.

## Environment Setup

Follow these steps to set up your environment and run the project.

### Step 1: Create a Conda Environment

First, create a new Conda environment named `obj_detect_track`. Open your terminal or command prompt and execute:

``` conda create --name obj_detect_track python=3.8 ```

This command creates a new environment with Python 3.8, which is compatible with all required packages.

Step 2: Activate the Environment
Activate the newly created environment:

```conda activate obj_detect_track```

Step 3: Install Required Packages
After activating the environment, install the necessary Python packages. This project's dependencies are listed in requirements.txt. To install them, use:

```pip install -r requirements.txt```

This command ensures you install the latest versions of the packages, including PyTorch, OpenCV, and NumPy, among others.

Running the Project
With the environment set up and all dependencies installed, you're ready to run the project. The main script is main.py, which accepts various command-line arguments for customizing the detection and tracking process.

To process a video, execute:

```python main.py --video_path /path/to/video.mp4 --model_path /path/to/yolov5s.pt --output_path /path/to/output --device cuda```

Replace /path/to/video.mp4 with the path to your video file, /path/to/yolov5s.pt with the path to the YOLOv5 model file, and /path/to/output with the directory where you want to save the output frames or video.

Additional Flags
The main.py script offers additional flags for customizing the detection and tracking:

--conf_thresh: Set the confidence threshold for detections.

--iou_thresh: Set the IoU threshold for Non-Maximum Suppression.

--save_video: If set, the output will be saved as a video file instead of individual frames.

--display: If set, the processed frames will be displayed in a window as the video is processed.
Notes


Ensure you have the YOLOv5 model file (yolov5s.pt) available. If not, download it from the official YOLOv5 GitHub repository.
The project is set up to use CUDA by default for GPU acceleration. If you're running on a system without a compatible GPU, change the --device argument to cpu.
