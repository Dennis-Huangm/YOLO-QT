# YOLO_Qt

A desktop GUI app built with `PyQt5 + Ultralytics YOLO` for object detection and training.

It supports:
- image detection
- video detection
- webcam detection
- dataset labeling flow
- train/val split generation
- YAML config generation
- model training from GUI

## Features

- Detection
  - Run inference on images, videos, and live camera streams
  - Display bounding boxes, labels, and confidence values in the UI
  - Save detection results to `runs/detect/...`
- Model loading
  - Use default `yolov8n.pt`
  - Load custom `.pt` weights
- Dataset prep
  - Launch integrated `labelimg`
  - Generate `train.txt` and `val.txt`
  - Generate `tmp.yaml` for training
- Training
  - Configure `epoch`, `batch`, `workers`, and `optimizer`
  - Run validation after training

## Requirements

- Python 3.8+
- Windows environment is the primary target in current scripts/UI flow
- Optional: NVIDIA GPU + CUDA for acceleration

## Install

```bash
git clone <your-repo-url>
cd YOLO_Qt
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Recommended (threaded UI):

```bash
python main_threaded.py
```

Legacy entry:

```bash
python main.py
```

## Quick Workflow

1. Initialize model in the UI (default or custom weights).
2. Open image/video or start camera detection.
3. Enable save option if you want output files.
4. For training:
   - select an `images` directory
   - split dataset to `train.txt` and `val.txt`
   - generate `tmp.yaml`
5. Set training parameters and start training.

## Project Structure

```text
YOLO_Qt/
|-- main_threaded.py      # Recommended entry (threaded)
|-- main.py               # Legacy entry
|-- mainwindow.py         # Generated UI Python code
|-- mainwindow.ui         # Qt Designer source
|-- requirements.txt
|-- train.py
|-- val.py
|-- labelimg/             # Labeling tool
|-- runs/                 # Detection/training outputs
|-- datasets/
`-- weights/
```

## Output Paths

- Detection output: `runs/detect/predict*`
- Training output: `runs/train/*`
- Validation output: `runs/val/*`

## Common Issues

- Failed to load weights
  - Verify the `.pt` path
  - First-time default model download requires network
- Video cannot open
  - Check video codec support in your OpenCV build
- Training fails
  - Verify paths in `tmp.yaml`, `train.txt`, `val.txt`
  - Verify class definitions are consistent

## Core Dependencies

See `requirements.txt` for full list. Main packages:
- `ultralytics`
- `opencv-python`
- `PyQt5`
- `numpy`
- `pyyaml`

## Notes

- The code sets `KMP_DUPLICATE_LIB_OK=TRUE` to reduce some MKL runtime conflicts.
- If you standardize on the threaded app, use `main_threaded.py` as your default startup script.
