"""
train.py
=========
Trains the CNN face-mask classifier (with_mask / without_mask) from the
Kaggle "omkargurav/face-mask-dataset" dataset.

This consolidates and cleans up the training logic found across
`Day_4_-Face_Mask_Detection.ipynb`, `Untitled5.ipynb` and `Untitled7.ipynb`
(the latter two were exploratory TensorFlow-installation notebooks whose
final working training code is reproduced here).

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 64 --img-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
import utils

logger = utils.get_logger(__name__)


def build_cnn_model(img_size: int):
    """
    Build and compile the CNN architecture used for binary mask
    classification (sigmoid output).
    """
    try:
        from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install it with "
            "`pip install tensorflow`."
        ) from exc

    model = Sequential(
        [
            Input(shape=(img_size, img_size, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history, output_path: Path) -> None:
    """Save accuracy/loss curves to disk (non-blocking, headless-safe)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend - safe on servers / CI
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping training plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history.get("accuracy", []), label="train_accuracy")
    axes[0].plot(history.history.get("val_accuracy", []), label="val_accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="train_loss")
    axes[1].plot(history.history.get("val_loss", []), label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved training history plot to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face-mask CNN classifier.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional local folder containing with_mask/without_mask sub-folders. "
        "If omitted, the dataset is downloaded automatically via kagglehub.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=config.CNN_MODEL_PATH,
        help="Where to save the trained .keras model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.error("scikit-learn is required. Install it with `pip install scikit-learn`.")
        return 1

    try:
        data_path = args.data_dir if args.data_dir else utils.download_face_mask_dataset()
        X, y = utils.load_image_dataset(data_path, img_size=args.img_size)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=config.RANDOM_STATE, stratify=y
        )
        logger.info("Train/Test split: %d / %d samples", len(X_train), len(X_test))

        model = build_cnn_model(args.img_size)
        model.summary(print_fn=logger.info)

        history = model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_test, y_test),
        )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info("Test Loss: %.4f | Test Accuracy: %.4f", loss, accuracy)

        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output_model))
        logger.info("Model saved to %s", args.output_model)

        plot_training_history(history, config.OUTPUTS_DIR / "training_history.png")

    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Training failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
