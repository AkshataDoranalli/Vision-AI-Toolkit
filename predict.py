"""
predict.py
===========
Command-line inference tool that consolidates every prediction workflow
found across the uploaded notebooks:

    * Haar Cascade face detection                (workshopday3, taskday2)
    * Haar Cascade + CNN mask classification      (Day_4_-Face_Mask_Detection)
    * YOLO general object detection                (Day_5_-_YOLO)
    * YOLO segmentation / pose estimation           (Day_5_-_YOLO)
    * YOLO custom face-mask detection (weights.pt)  (Day_5_-_YOLO)

Usage examples:
    # Detect faces only (Haar Cascade)
    python predict.py --task face --source webcam

    # CNN mask classification on a single image
    python predict.py --task mask-cnn --source path/to/photo.jpg

    # CNN mask classification on webcam
    python predict.py --task mask-cnn --source webcam

    # YOLO custom mask detector on an image
    python predict.py --task mask-yolo --source path/to/photo.jpg

    # YOLO general object detection on webcam
    python predict.py --task detect --source webcam

    # YOLO pose estimation on an image
    python predict.py --task pose --source path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import sys

import cv2

import config
import utils

logger = utils.get_logger(__name__)


# --------------------------------------------------------------------------
# Frame source helper (image file vs. webcam)
# --------------------------------------------------------------------------
def _is_webcam(source: str) -> bool:
    return source.lower() == "webcam"


def _run_on_webcam(process_frame_fn) -> None:
    """
    Open the default webcam and repeatedly call `process_frame_fn(frame)`
    (which must return the annotated frame to display) until the user
    presses 'q' or the stream ends.
    """
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera permissions/index.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame - stopping.")
                break

            annotated = process_frame_fn(frame)
            cv2.imshow("Prediction (press 'q' to quit)", annotated)

            if cv2.waitKey(1) & 0xFF == ord(config.QUIT_KEY):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _load_image(source: str):
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    return img


def _show_image(image_bgr, window_title: str = "Prediction") -> None:
    cv2.imshow(window_title, image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Task implementations
# --------------------------------------------------------------------------
def task_face(source: str) -> None:
    """Haar Cascade face detection (no mask classification)."""
    cascade = utils.get_face_cascade()

    def process(frame):
        faces = utils.detect_faces(cascade, frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_MASK, config.BOX_THICKNESS)
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 40),
            getattr(cv2, config.FONT),
            1,
            config.COLOR_NO_MASK,
            2,
        )
        return frame

    if _is_webcam(source):
        _run_on_webcam(process)
    else:
        image = _load_image(source)
        _show_image(process(image))


def task_mask_cnn(source: str) -> None:
    """Haar Cascade face detection + CNN mask/no-mask classification."""
    cascade = utils.get_face_cascade()
    model = utils.load_cnn_model()

    def process(frame):
        faces = utils.detect_faces(cascade, frame)
        for (x, y, w, h) in faces:
            face_crop = frame[y : y + h, x : x + w]
            try:
                label, confidence = utils.predict_mask_cnn(model, face_crop)
            except ValueError:
                continue  # empty crop at frame edge - skip safely
            utils.draw_label_box(frame, (x, y, w, h), label, confidence)
        return frame

    if _is_webcam(source):
        _run_on_webcam(process)
    else:
        image = _load_image(source)
        _show_image(process(image))


def task_mask_yolo(source: str) -> None:
    """Custom YOLO model (weights.pt) fine-tuned for face-mask detection."""
    model = utils.load_yolo_model(config.YOLO_MASK_WEIGHTS)

    def process(frame):
        results = model(frame, verbose=False)
        return results[0].plot()

    if _is_webcam(source):
        _run_on_webcam(process)
    else:
        image = _load_image(source)
        _show_image(process(image))


def task_detect(source: str) -> None:
    """General-purpose YOLO object detection."""
    model = utils.load_yolo_model(config.YOLO_DETECTION_WEIGHTS)

    def process(frame):
        resized = cv2.resize(frame, config.YOLO_WEBCAM_RESIZE)
        results = model(resized, verbose=False)
        return results[0].plot()

    if _is_webcam(source):
        _run_on_webcam(process)
    else:
        image = _load_image(source)
        results = model(image, verbose=False)
        _show_image(results[0].plot())


def task_segment(source: str) -> None:
    """YOLO instance segmentation."""
    model = utils.load_yolo_model(config.YOLO_SEGMENTATION_WEIGHTS)

    def process(frame):
        resized = cv2.resize(frame, config.YOLO_WEBCAM_RESIZE)
        results = model(resized, verbose=False)
        return results[0].plot()

    if _is_webcam(source):
        _run_on_webcam(process)
    else:
        image = _load_image(source)
        results = model(image, verbose=False)
        _show_image(results[0].plot())


def task_pose(source: str) -> None:
    """YOLO pose estimation."""
    model = utils.load_yolo_model(config.YOLO_POSE_WEIGHTS)

    def process(frame):
        resized = cv2.resize(frame, config.YOLO_WEBCAM_RESIZE)
        results = model(resized, verbose=False)
        return results[0].plot()

    if _is_webcam(source):
        _run_on_webcam(process)
    else:
        image = _load_image(source)
        results = model(image, verbose=False)
        _show_image(results[0].plot())


TASKS = {
    "face": task_face,
    "mask-cnn": task_mask_cnn,
    "mask-yolo": task_mask_yolo,
    "detect": task_detect,
    "segment": task_segment,
    "pose": task_pose,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the trained models.")
    parser.add_argument("--task", choices=sorted(TASKS.keys()), required=True)
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image file, or the literal string 'webcam'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        TASKS[args.task](args.source)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Prediction failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
