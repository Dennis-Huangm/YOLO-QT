# YOLO_Qt - YOLOv8 Object Detection GUI Application

A PyQt5-based graphical user interface for YOLOv8 object detection. This application provides an intuitive desktop environment for running object detection on images, videos, and live camera feeds, as well as training custom YOLO models.

## Features

### Object Detection
- **Image Detection**: Load and detect objects in single images (JPG, PNG)
- **Video Detection**: Process video files (MP4, FLV) with real-time detection
- **Camera Detection**: Live object detection from webcam/camera feed
- **Result Visualization**: Display detection results with bounding boxes, class labels, and confidence scores
- **Video Export**: Save detection results as MP4 videos

### Model Management
- **Pre-trained Models**: Support for YOLOv8 variants (nano, small, medium, large, etc.)
- **Custom Weights**: Load custom-trained model weights
- **Model Initialization**: Easy model loading and initialization

### Data Labeling & Training
- **Image Labeling**: Integrated LabelImg tool for dataset annotation
- **Data Splitting**: Automatic train/validation split with configurable ratios
- **YAML Configuration**: Auto-generate dataset configuration files
- **Model Training**: Train custom YOLO models with adjustable parameters:
  - Epochs (回合数)
  - Batch size (批次大小)
  - Number of workers (进程数)
  - Optimizer selection
  - Save options

### User Interface
- Real-time detection display with image preview
- Status bar for operation feedback
- Pause/Resume functionality for video processing
- Adjustable sliders for training hyperparameters
- Checkbox options for saving results

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd YOLO_Qt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 weights (optional, auto-downloads on first use):
```bash
# The application will automatically download yolov8n.pt on first run
# You can pre-download other models:
# yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
```

## Usage

### Running the Application

```bash
python main.py
```

### Basic Workflow

#### 1. Initialize Model
- Click **Start Model** (启动模型) button
- Optionally select custom weights via **Select Weights** (选择权重) button
- Wait for model initialization confirmation

#### 2. Object Detection

**For Images:**
- Click **Open File** (打开文件) button
- Select an image file
- View detection results in the preview area
- Check **Save Results** (保存结果) to save annotated images

**For Videos/Camera:**
- Click **Open File** (打开文件) for video files, or **Camera** (摄像头) for webcam
- Use **Pause** (暂停) to pause playback
- Use **Continue** (继续) to resume
- Use **Stop** (停止) to end detection
- Check **Save Results** (保存结果) to save output video

#### 3. Data Labeling & Training

**Label Images:**
1. Click **Select Folder** (选择文件夹) under labeling section
2. Enter class names in the text area
3. Click **Generate Classes** (生成类别) to create predefined classes
4. Click **Start Labeling** (开始标注) to launch LabelImg

**Prepare Dataset:**
1. Select the folder containing labeled images
2. Click **Split Data** (分割数据) to create train/val split
3. Click **Create YAML** (创建YAML) to generate dataset configuration

**Train Model:**
1. Select model architecture from dropdown
2. Adjust training parameters using sliders:
   - Epochs (训练的回合数)
   - Batch size (训练的批次大小)
   - Workers (载入数据的进程数)
3. Select optimizer
4. Check **Save Model** (保存模型) to save trained weights
5. Click **Train** (训练) to start training

## Project Structure

```
YOLO_Qt/
├── main.py              # Main application entry point
├── mainwindow.py        # PyQt5 UI definitions (auto-generated)
├── mainwindow.ui        # Qt Designer UI file
├── requirements.txt     # Python dependencies
├── train.py             # Training utilities
├── val.py               # Validation utilities
├── ex.py                # Example/utility script
├── labelimg/            # LabelImg annotation tool
├── datasets/            # Sample datasets
├── weights/             # Pre-trained model weights
├── runs/                # Detection/training output results
└── README.md            # This file
```

## Dependencies

- **opencv-python**: Computer vision operations
- **ultralytics**: YOLOv8 implementation
- **PyQt5**: GUI framework
- **numpy**: Numerical computations
- **pyyaml**: YAML configuration handling
- **pandas**: Data processing
- **Pillow**: Image processing
- **lxml**: XML processing for annotations

See `requirements.txt` for specific versions.

## Key Functions

### Detection
- `detect(img0)`: Run inference on a single frame/image
- `work()`: Main detection loop for videos/camera
- `cvimg_to_qtimg(cvimg)`: Convert OpenCV image to PyQt format

### Data Processing
- `split_data()`: Split dataset into train/validation sets
- `create_yaml()`: Generate YAML configuration for training
- `safe()`: Save predefined class definitions

### Model Training
- `train()`: Train custom YOLO model with specified parameters
- `epoch_f()`, `batch_f()`, `workers_f()`: Update training parameters

## Output

Detection results are saved in the `runs/detect/predict/` directory:
- Annotated images with bounding boxes
- Annotated videos with detection overlays
- Prediction logs with coordinates and class labels

Training results are saved in the `runs/train/` and `runs/val/` directories.

## Troubleshooting

### Model Loading Issues
- Ensure weights file exists or is accessible
- Check internet connection for auto-download
- Verify CUDA/GPU drivers if using GPU acceleration

### Detection Performance
- Use smaller models (nano, small) for faster inference
- Reduce video resolution if processing is slow
- Adjust batch size during training for memory constraints

### Labeling Issues
- Ensure image folder is named `images`
- Verify class count matches number of lines in class definition
- Check that LabelImg path is correctly configured

## Configuration

### Environment Variables
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # For Intel MKL compatibility
```

### Default Model
Default model is `yolov8n.pt` (nano variant). Change in code or select custom weights via GUI.

## License

This project uses YOLOv8 from Ultralytics. Please refer to the Ultralytics license for usage terms.

## Notes

- The application uses OpenCV's `grab()` and `retrieve()` methods for optimized video frame reading
- Results are automatically versioned with incrementing directory names
- Training supports custom optimizers and hyperparameter tuning
- GPU acceleration is automatically used if CUDA is available

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are correctly installed
3. Ensure input files are in supported formats
4. Check application logs in the status bar

---

**Version**: 1.0  
**Last Updated**: 2025
