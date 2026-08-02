"""
utils.py
=========
Shared utility functions used by train.py, predict.py and app.py.

Includes:
    - Logging setup
    - Dataset download / loading helpers (kagglehub + OpenCV)
    - CNN model loading & face-mask prediction helpers
    - YOLO model loading helpers
    - Haar Cascade face detector helper
    - Drawing helpers for bounding boxes / labels

All functions use explicit exception handling and pathlib for cross-platform
path safety, and never hardcode absolute/local paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

import config

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = __name__) -> logging.Logger:
    """Return a configured logger (idempotent - safe to call multiple times)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    return logger


logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------
def download_face_mask_dataset() -> Path:
    """
    Download (or reuse the cached copy of) the Kaggle face-mask dataset via
    kagglehub, and return the path to the folder that directly contains the
    `with_mask` / `without_mask` sub-folders.

    Raises:
        RuntimeError: if the dataset cannot be downloaded or the expected
            sub-folder layout is not found.
    """
    try:
        import kagglehub  # imported lazily so the rest of the project does
        # not hard-depend on kagglehub unless training is actually requested.
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the dataset. "
            "Install it with `pip install kagglehub`."
        ) from exc

    try:
        download_path = Path(kagglehub.dataset_download(config.KAGGLE_DATASET_SLUG))
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        raise RuntimeError(f"Failed to download dataset from Kaggle: {exc}") from exc

    # The `omkargurav/face-mask-dataset` dataset nests images inside a
    # "data" sub-folder. Fall back to the download root if that layout
    # ever changes upstream.
    candidate = download_path / "data"
    data_path = candidate if candidate.exists() else download_path

    missing = [c for c in config.CATEGORIES if not (data_path / c).exists()]
    if missing:
        raise RuntimeError(
            f"Expected category folders {missing} were not found under {data_path}. "
            "The dataset layout may have changed."
        )

    logger.info("Dataset ready at: %s", data_path)
    return data_path


def load_image_dataset(data_path: Path, img_size: int = config.IMG_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all images from `data_path/<category>/*` into a normalized numpy
    array (X) together with their integer labels (y).

    Args:
        data_path: folder containing one sub-folder per category.
        img_size: target width/height each image is resized to.

    Returns:
        (X, y) where X has shape (N, img_size, img_size, 3) and is scaled to
        [0, 1], and y has shape (N,) with integer class indices.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # tqdm is a soft dependency purely for a progress bar
        def tqdm(iterable: Iterable, **_kwargs):  # type: ignore
            return iterable

    images: List[np.ndarray] = []
    labels: List[int] = []

    for label_index, category in enumerate(config.CATEGORIES):
        folder_path = data_path / category
        if not folder_path.exists():
            raise FileNotFoundError(f"Category folder not found: {folder_path}")

        file_names = sorted(p.name for p in folder_path.iterdir() if p.is_file())
        for img_name in tqdm(file_names, desc=f"Loading '{category}'"):
            img_path = folder_path / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Skipping unreadable image: %s", img_path)
                continue
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label_index)

    if not images:
        raise RuntimeError(f"No images were loaded from {data_path}")

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)
    logger.info("Loaded dataset: X=%s y=%s", X.shape, y.shape)
    return X, y


# --------------------------------------------------------------------------
# CNN model helpers
# --------------------------------------------------------------------------
def load_cnn_model(model_path: Path = config.CNN_MODEL_PATH):
    """
    Load the trained Keras face-mask classifier.

    Raises:
        FileNotFoundError: if the model file does not exist.
        RuntimeError: if TensorFlow/Keras fails to load the model.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model not found at {model_path}. Run `python train.py` first, "
            "or place a trained `face_mask_model.keras` file in the models/ folder."
        )
    try:
        from tensorflow.keras.models import load_model
        model = load_model(str(model_path))
        logger.info("Loaded CNN model from %s", model_path)
        return model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load CNN model: {exc}") from exc


def preprocess_face_for_cnn(face_bgr: np.ndarray, img_size: int = config.IMG_SIZE) -> np.ndarray:
    """
    Resize / normalize / reshape a BGR face crop into the (1, H, W, 3) tensor
    expected by the CNN classifier.
    """
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Received an empty face crop for preprocessing.")
    resized = cv2.resize(face_bgr, (img_size, img_size))
    normalized = resized.astype(np.float32) / 255.0
    return np.reshape(normalized, (1, img_size, img_size, 3))


def predict_mask_cnn(model, face_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Run the CNN classifier on a single face crop.

    Returns:
        (label, confidence_percent) e.g. ("Mask", 97.3)
    """
    tensor = preprocess_face_for_cnn(face_bgr)
    raw_score = float(model.predict(tensor, verbose=0)[0][0])

    if raw_score > config.CNN_THRESHOLD:
        label = config.CNN_LABELS[1]  # "No Mask"
        confidence = raw_score * 100
    else:
        label = config.CNN_LABELS[0]  # "Mask"
        confidence = (1 - raw_score) * 100

    return label, confidence


# --------------------------------------------------------------------------
# Haar Cascade helper
# --------------------------------------------------------------------------
def get_face_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled frontal-face Haar Cascade classifier."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
    return cascade


def detect_faces(cascade: cv2.CascadeClassifier, frame_bgr: np.ndarray) -> np.ndarray:
    """Detect faces in a BGR frame and return an array of (x, y, w, h) boxes."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray,
        scaleFactor=config.HAAR_SCALE_FACTOR,
        minNeighbors=config.HAAR_MIN_NEIGHBORS,
        minSize=config.HAAR_MIN_SIZE,
    )


# --------------------------------------------------------------------------
# YOLO helpers
# --------------------------------------------------------------------------
def load_yolo_model(weights_path: Path):
    """
    Load a YOLO model (detection / segmentation / pose / custom) via
    ultralytics.

    Raises:
        FileNotFoundError: if a *custom* weights file does not exist locally.
        RuntimeError: if ultralytics fails to load the model.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO inference. Install it with "
            "`pip install ultralytics`."
        ) from exc

    # Pretrained, publicly-named weights (e.g. yolo26n.pt) are auto-downloaded
    # by ultralytics if missing, so we only hard-require local custom weights
    # such as weights.pt (the fine-tuned face-mask model).
    if weights_path.name == config.YOLO_MASK_WEIGHTS.name and not weights_path.exists():
        raise FileNotFoundError(
            f"Custom YOLO weights not found at {weights_path}. "
            "Place your trained weights.pt file inside models/."
        )

    try:
        model = YOLO(str(weights_path))
        logger.info("Loaded YOLO model from %s", weights_path)
        return model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load YOLO model '{weights_path}': {exc}") from exc


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def draw_label_box(
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    """Draw a colored bounding box + label/confidence text in-place on a frame."""
    x, y, w, h = box
    color = config.COLOR_MASK if label == config.CNN_LABELS[0] else config.COLOR_NO_MASK
    font = getattr(cv2, config.FONT)
    text = f"{label}: {confidence:.2f}%"

    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, config.BOX_THICKNESS)
    cv2.putText(
        frame_bgr,
        text,
        (x, max(y - 10, 0)),
        font,
        config.FONT_SCALE,
        color,
        config.FONT_THICKNESS,
    )


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convenience wrapper for matplotlib/Streamlit display (BGR -> RGB)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
