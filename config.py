"""
config.py
==========
Centralized configuration for the Face Mask Detection & YOLO Computer Vision project.

All paths are built with `pathlib.Path` so the project runs unmodified on
Windows, macOS, and Linux. Every "magic number" used across the codebase
(image size, thresholds, colors, hyperparameters, etc.) lives here so it can
be tuned from a single place.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Base directories (all relative to this file -> platform independent)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VALIDATION_DIR = DATA_DIR / "validation"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure runtime directories exist (safe no-op if they already do)
for _directory in (MODELS_DIR, DATA_DIR, TRAIN_DIR, TEST_DIR, VALIDATION_DIR, OUTPUTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset configuration
# --------------------------------------------------------------------------
# Kaggle dataset used to train the CNN mask classifier (downloaded via kagglehub)
KAGGLE_DATASET_SLUG = "omkargurav/face-mask-dataset"

# Category names must match the sub-folder names inside the dataset ("data/")
CATEGORIES = ["with_mask", "without_mask"]

# Index -> human readable label used everywhere for display purposes.
# NOTE: index 0 == "with_mask", index 1 == "without_mask" (see CATEGORIES above)
CNN_LABELS = {0: "Mask", 1: "No Mask"}

# --------------------------------------------------------------------------
# CNN model configuration (Keras face-mask classifier)
# --------------------------------------------------------------------------
IMG_SIZE = 64  # width/height the CNN expects (64x64x3)
CNN_MODEL_PATH = MODELS_DIR / "face_mask_model.keras"

# Training hyperparameters
EPOCHS = 15
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 1e-3

# Decision threshold for the sigmoid output of the CNN.
# score > CNN_THRESHOLD  -> "without_mask" (label index 1)
# score <= CNN_THRESHOLD -> "with_mask"    (label index 0)
CNN_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Haar Cascade (classical face detector used with the CNN classifier)
# --------------------------------------------------------------------------
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (60, 60)

# --------------------------------------------------------------------------
# YOLO model configuration
# --------------------------------------------------------------------------
# Custom-trained YOLO model (fine-tuned on face-mask data) - included in models/
YOLO_MASK_WEIGHTS = MODELS_DIR / "weights.pt"

# Pretrained general-purpose object detection / pose models.
# These are downloaded automatically by the `ultralytics` package on first
# use if they are not already present in models/. Placing them in MODELS_DIR
# keeps the whole project self-contained.
YOLO_DETECTION_WEIGHTS = MODELS_DIR / "yolo26n.pt"
YOLO_SEGMENTATION_WEIGHTS = MODELS_DIR / "yolo26n-seg.pt"
YOLO_POSE_WEIGHTS = MODELS_DIR / "yolo26n-pose.pt"

# Resolution the webcam frame is resized to before running YOLO inference
YOLO_WEBCAM_RESIZE = (1080, 600)

# --------------------------------------------------------------------------
# Drawing / display configuration (BGR color tuples for OpenCV)
# --------------------------------------------------------------------------
COLOR_MASK = (0, 255, 0)      # green
COLOR_NO_MASK = (0, 0, 255)   # red
FONT = "FONT_HERSHEY_SIMPLEX"  # resolved via getattr(cv2, FONT) in utils.py
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
WEBCAM_INDEX = 0
QUIT_KEY = "q"
SAVE_KEY = "s"
LOG_LEVEL = "INFO"
