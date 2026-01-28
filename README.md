# Geometry-Aware 3D Traffic Analysis from Monocular Video

This project implements a geometry-aware 3D traffic analysis pipeline for monocular video recorded using a fixed camera. The system detects and tracks vehicles, reconstructs their positions in world coordinates, and estimates physically interpretable quantities such as vehicle speed, inter-vehicle distance and overtaking behaviour.

---

## Features

- Vehicle detection and multi-object tracking from monocular video
- Camera calibration and image-to-world coordinate transformation
- 3D bounding box generation for vehicle representation
- Real-world speed estimation (km/h)
- Inter-vehicle distance and spatial relationship estimation
- Comparative evaluation of multiple geometric estimation approaches
- Export of processed results for offline analysis

---

## Project Structure

```text
Thesis/
├── data/                                # Input data, models, and calibration files
│   ├── 3dbb_results.json
│   ├── best.pt
│   ├── calibration-lookup-table.npy
│   ├── coordinate_mapping_2030.json
│   ├── gopro_calibration_fisheye.npz
│   ├── refpts1.png
│   ├── refpts1.txt
│   ├── refpts2.png
│   ├── refpts2.txt
│   ├── sample_image.png
│   ├── yolov8n-seg.pt
│   └── yolov8n.pt
│
├── output/                              # Generated outputs and evaluation results
│   ├── corner_points.json
│   ├── detections_kitti.txt
│   └── full_comparison_report.json
│
├── src/                                 # Core thesis implementation
│   ├── bbox_3d_generator.py
│   ├── compare_all_approaches.py
│   ├── distance_analyzer.py
│   ├── fallback.py
│   ├── interactive_3d_bbox.py
│   └── speed_estimator.py
│
├── utils/                               # Supporting tools and utilities
│   ├── .gitkeep                      
│   └── calibration_tool/
│       ├── .gitignore
│       ├── README.md
│       ├── main.py
│       ├── requirements.txt
│       ├── stabilization.py
│       ├── Beispieldaten/            
│       │   ├── .gitkeep
│       │   ├── Max_Pla.txt
│       │   └── example-street.png
│       └── utils/
│           ├── calibration.py
│           ├── roi.py
│           └── selectPoints.py
│
├── .gitignore
└── requirements.txt
```

---

## Prerequisites

- Python 3.8 or higher
- macOS, Windows or Linux (tested on macOS)
- Monocular traffic video recorded using a fixed camera
- Camera calibration parameters corresponding to the recording setup

A GPU is recommended for faster inference but is not strictly required.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Thesis.git
cd Thesis
```

Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

- Place input videos, calibration files, and lookup tables in the `data/` directory.
- Ensure that paths to input data and outputs are correctly specified within the corresponding scripts.
- If required, download and place trained model weights (e.g., YOLO checkpoints) in the appropriate location.

---

## Usage

All executable scripts are located in the `src/` directory and should be run from the repository root.

### Generate 3D vehicle representations

```bash
python src/bbox3d_generator.py \
    --image data/sample_image.png \
    --lookup data/calibration_lookup.npy
```

### Interactive refinement (optional)

```bash
python src/interactive_3d_bbox.py \
    --image data/sample_image.png \
    --lookup data/calibration_lookup.npy
```

### Estimate vehicle speed

```bash
python src/speed_estimator.py \
    --lookup data/calibration_lookup.npy \
    --video data/input_video.mp4 \
    --output output/speed_result.mp4
```

### Analyze inter-vehicle distances and overtaking behaviour

```bash
python src/distance_analyzer.py \
    --lookup data/calibration_lookup.npy \
    --video data/input_video.mp4 \
    --output output/distance_result.mp4
```

### Compare geometric approaches

```bash
python src/compare_all_approaches.py \
    --video data/input_video.mp4 \
    --mapping data/pixel_to_world_mapping.json \
    --vehicle-model models/vehicle_model.pt \
    --wheel-model models/wheel_seg_model.pt \
    --max-frames 300 --show
```

### Fallback strategies

```bash
python src/speed_pipeline.py \
    --video data/input_video.mp4 \
    --vehicle-model models/vehicle.pt \
    --wheel-model models/wheel_seg.pt \
    --calibration data/calibration.npz \
    --mapping data/mapping.json \
    --output-csv output/speeds.csv
```


---

## Data and Outputs

- `data/` contains input videos, camera calibration parameters, lookup tables, and intermediate data files.
- `output/` contains generated results such as annotated videos, plots, logs, and quantitative evaluation outputs.

To keep the repository lightweight, large data and output files may be excluded from version control via `.gitignore`.

---

## Author

Shalikha Rajesh  
Master’s Thesis Project



