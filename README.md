# 🎯 Vision AI Toolkit — Face Mask Detection \& YOLO Computer Vision

A production-ready computer vision project combining a custom-trained **CNN face-mask
classifier** (Keras/TensorFlow), a **Haar Cascade face detector**, and multiple
**YOLO models** (object detection, instance segmentation, pose estimation, and a
custom fine-tuned face-mask detector) into one clean, GitHub-ready codebase.

This repository consolidates a series of workshop/learning notebooks into
reusable, tested, production-style Python modules.

\---

## 📋 Overview

The project supports two complementary approaches to face-mask detection:

1. **Classical + Deep Learning pipeline** — OpenCV Haar Cascade locates faces,
then a custom CNN (trained on the
[face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset))
classifies each face as `Mask` / `No Mask`.
2. **End-to-end YOLO pipeline** — A YOLO model fine-tuned directly on face-mask
data (`weights.pt`) detects and classifies masks in a single pass.

In addition, the project exposes general-purpose YOLO capabilities used during
the original workshops: object detection, instance segmentation, and pose
estimation — runnable on images or a live webcam feed, and through a Streamlit
web UI.

\---

## ✨ Features

* 🟢 Real-time face detection (OpenCV Haar Cascade)
* 😷 CNN-based face-mask classification (Keras `.keras` model)
* 🎯 Custom YOLO face-mask detector (`weights.pt`)
* 📦 General YOLO object detection
* 🧩 YOLO instance segmentation
* 🕺 YOLO pose estimation
* 🖥️ CLI inference tool (`predict.py`) for images and webcam streams
* 🌐 Streamlit web app (`app.py`) — upload an image or use your camera
* 🏋️ One-command training pipeline (`train.py`) with automatic Kaggle dataset download
* 🛠️ Centralized, typed configuration (`config.py`) — no hardcoded paths
* 🧱 Reusable utility layer (`utils.py`) with full exception handling
* 🖥️ Fully platform-independent (Windows / macOS / Linux) via `pathlib`

\---

## 📁 Folder Structure

```
Project/
│
├── app.py                 # Streamlit web application
├── train.py                # CNN training pipeline (CLI)
├── predict.py               # CLI inference tool (image / webcam)
├── requirements.txt         # Pinned, compatible dependency versions
├── README.md
├── .gitignore
├── config.py                # All paths \& hyperparameters (single source of truth)
├── utils.py                 # Shared helper functions \& classes
│
├── models/
│   ├── face\_mask\_model.keras   # Trained CNN mask classifier
│   ├── weights.pt               # Custom YOLO face-mask detector
│   ├── yolo26n.pt               
│   └── yolo26n-pose.pt          
│
├── data/
│   ├── train/
│   ├── test/
│   └── validation/
│
├── notebooks/               # Original exploratory notebooks (cleaned, renamed)
│   ├── Day\_2\_Workshop\_OpenCV\_Basics.ipynb
│   ├── Day\_3\_Workshop\_Video\_and\_Haar\_Cascade.ipynb
│   ├── Day\_4\_Face\_Mask\_Detection.ipynb
│   ├── Day\_5\_YOLO.ipynb
│   ├── Task\_Day\_2\_Webcam\_Face\_Detection.ipynb
│   └── Task\_Day\_4\_TensorFlow\_Setup\_Notes.ipynb
│
└── outputs/                 # Training plots, saved predictions, etc.
```

\---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Vision-AI-Toolkit

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> \*\*Note:\*\* TensorFlow and Ultralytics wheels are platform/Python-version
> specific. If installation fails, install a version of Python within the
> 3.9–3.12 range (TensorFlow does not yet ship wheels for every new Python
> release), or check the \[TensorFlow install guide](https://www.tensorflow.org/install)
> for your OS.

\---

## 📦 Dataset Setup

The CNN classifier is trained on the Kaggle
[`omkargurav/face-mask-dataset`](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).

`train.py` downloads it automatically via `kagglehub` 
(`\~/.kaggle/kaggle.json` or the `KAGGLE\_USERNAME` / `KAGGLE\_KEY` environment
variables).

Alternatively, pass a local dataset folder that already contains
`with\_mask/` and `without\_mask/` sub-folders:

```bash
python train.py --data-dir path/to/local/dataset
```

\---

## 🏋️ Training

```bash
python train.py --epochs 15 --batch-size 32 --img-size 64
```

This will:

1. Download (or reuse) the dataset.
2. Load and normalize all images.
3. Split into train/test sets.
4. Build and train the CNN.
5. Evaluate on the held-out test set.
6. Save the trained model to `models/face\_mask\_model.keras`.
7. Save an accuracy/loss plot to `outputs/training\_history.png`.

All hyperparameters are configurable via CLI flags or by editing `config.py`.

\---

## 🔍 Inference

### CLI (`predict.py`)

```bash
# Haar Cascade face detection only
python predict.py --task face --source webcam

# CNN face-mask classification on an image
python predict.py --task mask-cnn --source photo.jpg

# CNN face-mask classification on webcam
python predict.py --task mask-cnn --source webcam

# Custom YOLO face-mask detector
python predict.py --task mask-yolo --source photo.jpg

# General YOLO object detection
python predict.py --task detect --source webcam

# YOLO instance segmentation
python predict.py --task segment --source photo.jpg

# YOLO pose estimation
python predict.py --task pose --source photo.jpg
```

Press **`q`** to close any webcam window.

### Web app (`app.py`)

```bash
streamlit run app.py
```

Then open the printed local URL, choose a detection mode in the sidebar,
upload an image (or take a camera snapshot), and click **Run detection**.

\---



## 📊 Results

|Model|Task|Metric|
|-|-|-|
|CNN (`face\_mask\_model.keras`)|Mask / No-Mask classification|Test accuracy: 
|YOLO (`weights.pt`)|Face-mask detection|mAP: 
Re-run `train.py` to regenerate up-to-date metrics and the plot in
`outputs/training\_history.png`.

\---

## 🚀 Future Improvements

* Add data augmentation (rotation, brightness, occlusion) to improve CNN robustness.
* Fine-tune YOLO on a larger, more diverse mask dataset (varied angles, lighting, mask types).
* Add automated evaluation (precision/recall/mAP) reporting to `outputs/`.
* Containerize with Docker for one-command deployment.
* Add unit tests (`pytest`) for `utils.py` helper functions.
* Support batch video file processing (not just webcam/single image).
* Deploy the Streamlit app to Streamlit Community Cloud / Hugging Face Spaces.

\---

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
You are free to use, modify, and distribute it with attribution.

\---

## 🙏 Acknowledgements

* Dataset: [omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) (Kaggle)
* Object detection framework: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* Built on top of exploratory workshop notebooks covering OpenCV fundamentals,
Haar Cascades, CNNs, and YOLO.

