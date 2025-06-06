# Goal Detection in Football Using YOLOv8 and Homography

This project implements a system to **automatically detect goals in football matches** by combining ball detection with YOLOv8, spatial reasoning via homography, and goalpost detection from visual features.

The system is designed for use with pre-recorded videos and can be adapted for real-time applications.

## ⚽ Objective

The goal is to reliably detect when the ball **crosses the goal line**, even in the presence of:
- Camera distortion or limited depth perception
- Partial occlusion or motion blur
- Missed detections from the ball detector

This is **not a general tracking system**; it is explicitly focused on determining **goal-scoring events**.

## 🛠 Technologies Used

- [YOLOv8](https://github.com/ultralytics/ultralytics): object detection engine used to detect the ball
- Custom-trained ball detection model (trained on blurred/fuzzy ball samples)
- OpenCV: image processing, Hough Transform, homography estimation, and video generation
- Geometric analysis: goalposts detected using white vertical and horizontal lines
- Homography: perspective mapping to a flat goal plane for determining when the ball crosses the line


## 🧠 How It Works

### 1. Ball Detection (YOLOv8)
- A YOLOv8 model trained specifically to recognize footballs under difficult conditions (motion blur, low contrast, side views).
- Region of interest (ROI) is dynamically adjusted to focus search based on previous detections.
- If detection fails, a fallback mechanism attempts to recover from the last known location.

### 2. Goalpost Detection
- Goalposts are inferred using white lines detected via Hough Transform.
- A valid goal structure is composed of two vertical posts and a horizontal top bar.
- If goalposts are not detected in a frame, previous frame detections are reused.

### 3. Homography and Goal Validation
- Once goalposts are located, a homography is computed to map their 2D image location into a flat 2x1 meter reference space.
- The ball's position is transformed using this matrix.
- If the ball’s mapped location crosses the interior plane of the goal, it is counted as a goal.

### 4. Video Output
- The system outputs an annotated video, highlighting goal areas, ball trajectory, and detected goals.

## 🧪 Custom Ball Detector

A YOLOv8 model was trained using real match footage, including:
- Blurred, occluded, or distant ball appearances
- Side and overhead camera angles

▶️ Running the Pipeline
Install dependencies:

!pip install  ultralytics opencv-python-headless scikit-learn
!pip install inference supervision

Change the video input, start and finish frame


max_frames = 500  # ajustá este número según lo que necesites
start_frame = 200 # Define el número de frame donde quieres empezar
filename='/content/video3.webm.mkv'


Execute the scrypt

python detectGoal.py

🙏 Credits
Ultralytics: for YOLOv8

Roboflow: for model training and dataset management
