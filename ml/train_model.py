from datetime import datetime
from pathlib import Path
import argparse
import struct
import sys

import cv2
import joblib
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.preprocess import IMAGE_SIZE, preprocess_gray_image

MNIST_DIR = BASE_DIR / "data" / "mnist"
KAGGLE_DIR = BASE_DIR / "data" / "handwritten_digits"
CUSTOMIZE_DIR = BASE_DIR / "data" / "customize_data"
MODEL_DIR = BASE_DIR / "ml"

TRAIN_IMAGES = MNIST_DIR / "train-images.idx3-ubyte"
TRAIN_LABELS = MNIST_DIR / "train-labels.idx1-ubyte"
TEST_IMAGES = MNIST_DIR / "t10k-images.idx3-ubyte"
TEST_LABELS = MNIST_DIR / "t10k-labels.idx1-ubyte"

MODEL_CONFIGS = {
    "mnist": {
        "display_name": "Model 1 - MNIST ubyte",
        "dataset": "MNIST ubyte data",
        "path": MODEL_DIR / "digit_model_mnist.joblib",
    },
    "kaggle": {
        "display_name": "Model 2 - Kaggle folder images",
        "dataset": "Kaggle folder image data",
        "path": MODEL_DIR / "digit_model_kaggle.joblib",
    },
    "combined": {
        "display_name": "Model 3 - Combined MNIST + Kaggle",
        "dataset": "MNIST ubyte + Kaggle folder image data",
        "path": MODEL_DIR / "digit_model_combined.joblib",
    },
}

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
RANDOM_STATE = 42


def read_idx_images(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing image file: {file_path}")

    with open(file_path, "rb") as file:
        magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid image file magic number: {magic}")

        image_data = np.frombuffer(file.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        images = images.astype("float32") / 255.0

        return images.reshape(num_images, rows * cols)


def read_idx_labels(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing label file: {file_path}")

    with open(file_path, "rb") as file:
        magic, num_labels = struct.unpack(">II", file.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid label file magic number: {magic}")

        labels = np.frombuffer(file.read(), dtype=np.uint8)

        if len(labels) != num_labels:
            raise ValueError(f"Expected {num_labels} labels, found {len(labels)}")

        return labels


def load_mnist_data():
    print("Loading MNIST ubyte dataset...")

    X_train = read_idx_images(TRAIN_IMAGES)
    y_train = read_idx_labels(TRAIN_LABELS)
    X_test = read_idx_images(TEST_IMAGES)
    y_test = read_idx_labels(TEST_LABELS)

    print(f"MNIST train: {X_train.shape}, test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def find_labeled_image_paths(dataset_dir, dataset_name, missing_ok=False):
    if not dataset_dir.exists():
        if missing_ok:
            return [], np.array([], dtype=np.uint8)
        raise FileNotFoundError(f"Missing {dataset_name} folder: {dataset_dir}")

    image_paths = []
    labels = []

    for label_dir in sorted(dataset_dir.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.isdigit():
            continue

        label = int(label_dir.name)

        if label < 0 or label > 9:
            continue

        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.append(image_path)
                labels.append(label)

    if not image_paths:
        if missing_ok:
            return [], np.array([], dtype=np.uint8)
        raise ValueError(
            f"No {dataset_name} images found. Expected folders like "
            f"{dataset_dir}/0/*.jpg"
        )

    return image_paths, np.array(labels, dtype=np.uint8)


def find_kaggle_image_paths():
    return find_labeled_image_paths(KAGGLE_DIR, "Kaggle")


def load_kaggle_data():
    print("Loading Kaggle folder image dataset...")

    kaggle_paths, kaggle_labels = find_kaggle_image_paths()
    custom_paths, custom_labels = find_labeled_image_paths(
        CUSTOMIZE_DIR,
        "custom feedback",
        missing_ok=True,
    )

    image_paths = kaggle_paths + custom_paths
    labels = kaggle_labels

    if len(custom_paths) > 0:
        labels = np.concatenate([kaggle_labels, custom_labels])
        print(f"Including {len(custom_paths)} custom feedback images.")

    features = []

    for index, image_path in enumerate(image_paths, start=1):
        gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        features.append(preprocess_gray_image(gray_image))

        if index % 2500 == 0:
            print(f"Processed {index}/{len(image_paths)} folder images...")

    X = np.array(features, dtype=np.float32)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Kaggle/custom train: {X_train.shape}, test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def create_model():
    return ExtraTreesClassifier(
        n_estimators=160,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )


def train_and_save(model_id, X_train, X_test, y_train, y_test):
    config = MODEL_CONFIGS[model_id]

    print(f"\nTraining {config['display_name']}...")
    model = create_model()
    model.fit(X_train, y_train)

    print(f"Testing {config['display_name']}...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:")
    print(f"{accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    model_bundle = {
        "model": model,
        "model_id": model_id,
        "display_name": config["display_name"],
        "dataset": config["dataset"],
        "accuracy": round(float(accuracy * 100), 2),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
        "image_size": IMAGE_SIZE,
        "classes": list(range(10)),
        "confidence_type": "ExtraTrees predicted class probability",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    joblib.dump(model_bundle, config["path"])

    print(f"\nSaved {config['display_name']} at: {config['path']}")


def build_datasets():
    datasets = {}

    try:
        datasets["mnist"] = load_mnist_data()
    except Exception as exc:
        print(f"Skipping MNIST model: {exc}")

    try:
        datasets["kaggle"] = load_kaggle_data()
    except Exception as exc:
        print(f"Skipping Kaggle model: {exc}")

    if "mnist" in datasets and "kaggle" in datasets:
        print("Building combined MNIST + Kaggle dataset...")
        mnist_train, mnist_test, mnist_y_train, mnist_y_test = datasets["mnist"]
        kaggle_train, kaggle_test, kaggle_y_train, kaggle_y_test = datasets["kaggle"]

        datasets["combined"] = (
            np.vstack([mnist_train, kaggle_train]),
            np.vstack([mnist_test, kaggle_test]),
            np.concatenate([mnist_y_train, kaggle_y_train]),
            np.concatenate([mnist_y_test, kaggle_y_test]),
        )
    else:
        print("Skipping combined model because one of the datasets is unavailable.")

    return datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train handwritten digit recognition models."
    )
    parser.add_argument(
        "--model",
        choices=["all", "mnist", "kaggle", "combined"],
        default="all",
        help="Choose which model to train.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    datasets = build_datasets()

    if args.model == "all":
        model_ids = ["mnist", "kaggle", "combined"]
    else:
        model_ids = [args.model]

    for model_id in model_ids:
        if model_id not in datasets:
            print(f"\nCannot train {model_id}: dataset was not prepared.")
            continue

        train_and_save(model_id, *datasets[model_id])


if __name__ == "__main__":
    main()
