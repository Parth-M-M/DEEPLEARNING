# DEEP LEARNING PR 3

**Institute:** Red & White Skill Education
**Subject:** Deep Learning
**Project Duration:** 3 Days (2 Hours per Day)

## Project Overview
This project implements a comprehensive computer vision pipeline that transitions from classical image-processing operations to modern deep-learning-based detection methodologies. Designed to mirror real-world industry applications like edge-deployed smart-surveillance or retail-analytics systems, the pipeline combines classical pre-processing for noise reduction with deep learning models (YuNet and YOLO) for accurate real-time identification. 

## Frameworks and Dependencies
* Python 3.x utilizing Jupyter Notebook
* OpenCV (`cv2`) 5.0.0
* Ultralytics YOLOv8
* NumPy
* Matplotlib

### Installation
To install the required dependencies, run the following command:
`pip install opencv-python==5.0.0.* opencv-contrib-python==5.0.0.* ultralytics matplotlib numpy`

## Datasets and Models
* **Static Image Pack**: 6–10 sample images stored in the `/data/images` folder, including OpenCV sample data (e.g., `lena.jpg`, `baboon.jpg`, `coins.png`) and supplemented with custom photos.
* **Face Detector Weights**: `face_detection_yunet_2023mar.onnx` (OpenCV 5.0's bundled YuNet detector) stored in the `/data/models` folder.
* **Object Detector Weights**: `yolov8n.pt` (Ultralytics YOLOv8-Nano, pretrained on the 80-class COCO dataset).
* **Live Input**: Real-time video stream captured via a laptop webcam using `cv2.VideoCapture(0)`.

## Project Tasks and Methodology

* **Task 1: Environment Setup, Image I/O & Morphological Operations**
  Explores classical pixel-level operations including BGR to Grayscale conversion, binary thresholding, and morphological operations (Erosion, Dilation, Opening, Closing). This includes testing different structuring element shapes (`MORPH_RECT`, `MORPH_ELLIPSE`, `MORPH_CROSS`) and sizes to filter noise.
* **Task 2: Bitwise Operations & Image Histograms**
  Applies Bitwise AND, OR, XOR, and NOT for masked overlays and region extraction. Includes the calculation and plotting of grayscale and per-channel color histograms, as well as brightness and contrast adjustments utilizing `alpha` and `beta` parameters (`cv2.convertScaleAbs`).
* **Task 3: Face Detection with YuNet (OpenCV 5.0)**
  Implements the CNN-based YuNet face detector on both static images and live webcam feeds to extract bounding boxes, 5-point facial landmarks, and confidence scores. Features score-threshold tuning and a practical privacy-blur application using `cv2.GaussianBlur` on detected face regions.
* **Task 4: Object Detection with YOLO**
  Deploys the YOLOv8n single-stage detector to identify 80 COCO object classes. Analyzes real-time FPS, tests confidence and IoU thresholds (observing Non-Max Suppression behavior), and summarizes per-class detection counts.
* **Task 5: Integrated Pipeline & Final Comparison**
  Combines the YuNet face detector (green boxes) and the YOLO object detector (red boxes with labels) onto a single output frame. Evaluates the accuracy-to-compute trade-off by benchmarking FPS for each model individually and within the combined pipeline.

## Plots and Results



* **Morphology Grid**: `![Morphology Operations](data/content/data/plots/task1_3_morphological_ops.png)`
* **Histogram Comparison**: `![Histograms](data/content/data/plots/task2_3_histograms.png)`
* **Face Detection Comparison**: `![YuNet Detections](plots/face_detection_comparison.png)`
* **YOLO Detections**: `![YOLOv8 Results](data/content/data/plots/task4_1_yolo_detection.jpg)`
* **FPS Comparison**: `![FPS Benchmarking](data/content/data/plots/webcamdetect.png)`
* **Results Table**: `![Final Comparison Table](plots/results_table.png)`

## Video Presentation

* **Video Link**: [Insert Google Drive or Unlisted YouTube Link Here]
* The demonstration video (5–10 minutes) includes a webcam picture-in-picture overlay and screen capture. It features a verbal explanation of core concepts (morphology filters, histograms, YuNet mechanisms, and YOLO architecture) alongside live webcam demonstrations of the detection pipelines and FPS evaluations.
