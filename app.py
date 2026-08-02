"""
app.py
=======
Streamlit web application that exposes every detection mode built in this
project through a simple UI:

    * Haar Cascade face detection
    * CNN (Keras) face-mask classification
    * Custom YOLO face-mask detection
    * General YOLO object detection / segmentation / pose estimation

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

import config
import utils

logger = utils.get_logger(__name__)

st.set_page_config(page_title="Vision AI Toolkit", page_icon="🎯", layout="centered")


# --------------------------------------------------------------------------
# Cached resource loaders (avoids reloading models on every rerun)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading face cascade...")
def _get_cascade():
    return utils.get_face_cascade()


@st.cache_resource(show_spinner="Loading CNN mask classifier...")
def _get_cnn_model():
    return utils.load_cnn_model()


@st.cache_resource(show_spinner="Loading YOLO model...")
def _get_yolo_model(weights_path):
    return utils.load_yolo_model(weights_path)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image (as returned by Streamlit uploaders) to BGR for OpenCV."""
    import cv2

    rgb_array = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def _run_face_detection(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    cascade = _get_cascade()
    faces = utils.detect_faces(cascade, frame_bgr)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), config.COLOR_MASK, config.BOX_THICKNESS)
    st.caption(f"Detected {len(faces)} face(s).")
    return frame_bgr


def _run_mask_cnn(frame_bgr: np.ndarray) -> np.ndarray:
    cascade = _get_cascade()
    model = _get_cnn_model()
    faces = utils.detect_faces(cascade, frame_bgr)

    for (x, y, w, h) in faces:
        face_crop = frame_bgr[y : y + h, x : x + w]
        try:
            label, confidence = utils.predict_mask_cnn(model, face_crop)
        except ValueError:
            continue
        utils.draw_label_box(frame_bgr, (x, y, w, h), label, confidence)

    st.caption(f"Detected {len(faces)} face(s).")
    return frame_bgr


def _run_yolo(frame_bgr: np.ndarray, weights_path) -> np.ndarray:
    model = _get_yolo_model(weights_path)
    results = model(frame_bgr, verbose=False)
    return results[0].plot()


MODE_HANDLERS = {
    "Face Detection (Haar Cascade)": lambda frame: _run_face_detection(frame),
    "Face Mask Classification (CNN)": lambda frame: _run_mask_cnn(frame),
    "Face Mask Detection (Custom YOLO)": lambda frame: _run_yolo(frame, config.YOLO_MASK_WEIGHTS),
    "Object Detection (YOLO)": lambda frame: _run_yolo(frame, config.YOLO_DETECTION_WEIGHTS),
    "Instance Segmentation (YOLO)": lambda frame: _run_yolo(frame, config.YOLO_SEGMENTATION_WEIGHTS),
    "Pose Estimation (YOLO)": lambda frame: _run_yolo(frame, config.YOLO_POSE_WEIGHTS),
}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main() -> None:
    st.title("🎯 Vision AI Toolkit")
    st.write(
        "Upload an image or take a snapshot to run face detection, "
        "face-mask classification, object detection, segmentation, or pose estimation."
    )

    mode = st.sidebar.selectbox("Detection mode", list(MODE_HANDLERS.keys()))
    input_method = st.sidebar.radio("Input source", ["Upload image", "Camera snapshot"])

    image = None
    if input_method == "Upload image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            image = Image.open(camera_file)

    if image is None:
        st.info("Please provide an image to begin.")
        return

    st.image(image, caption="Input image", use_container_width=True)

    if st.button("Run detection", type="primary"):
        try:
            frame_bgr = _pil_to_bgr(image)
            with st.spinner("Running inference..."):
                annotated_bgr = MODE_HANDLERS[mode](frame_bgr)
            st.image(utils.bgr_to_rgb(annotated_bgr), caption="Result", use_container_width=True)
        except FileNotFoundError as exc:
            st.error(f"Required model file is missing: {exc}")
        except RuntimeError as exc:
            st.error(f"Inference failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors to the user
            logger.exception("Unexpected error during inference")
            st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
