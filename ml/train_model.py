from datetime import datetime
from pathlib import Path
import argparse
import json
import struct
import sys

import cv2
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
VISUALIZATION_DIR = BASE_DIR / "visualization"

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
    "custom_combined": {
        "display_name": "Model 4 - Handwritten + Custom Feedback",
        "dataset": "Kaggle folder image data + custom feedback data",
        "path": MODEL_DIR / "digit_model_custom_combined.joblib",
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


def load_folder_data(image_paths, labels, dataset_name):
    features = []

    for index, image_path in enumerate(image_paths, start=1):
        gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        features.append(preprocess_gray_image(gray_image))

        if index % 2500 == 0:
            print(f"Processed {index}/{len(image_paths)} {dataset_name} images...")

    X = np.array(features, dtype=np.float32)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"{dataset_name} train: {X_train.shape}, test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def load_kaggle_data():
    print("Loading Kaggle folder image dataset...")

    image_paths, labels = find_kaggle_image_paths()

    return load_folder_data(image_paths, labels, "Kaggle")


def load_custom_combined_data():
    print("Loading handwritten digit folder + custom feedback dataset...")

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
    else:
        print("No custom feedback images found. Training with handwritten_digits only.")

    return load_folder_data(image_paths, labels, "Handwritten/custom")


def create_model():
    return ExtraTreesClassifier(
        n_estimators=160,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )


def save_confusion_matrix_chart(matrix, labels, output_path, title, normalize=False):
    display_matrix = matrix.astype(float)

    if normalize:
        row_totals = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_totals,
            out=np.zeros_like(display_matrix),
            where=row_totals != 0,
        )

    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(display_matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set(
        title=title,
        xlabel="Predicted digit",
        ylabel="Actual digit",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )

    threshold = display_matrix.max() / 2 if display_matrix.size else 0
    for row_index in range(display_matrix.shape[0]):
        for column_index in range(display_matrix.shape[1]):
            raw_value = display_matrix[row_index, column_index]
            if normalize:
                text_value = format(raw_value, ".2f")
            else:
                text_value = str(int(raw_value))
            axis.text(
                column_index,
                row_index,
                text_value,
                ha="center",
                va="center",
                color="white" if raw_value > threshold else "#111827",
                fontsize=9,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_per_class_metrics_chart(report, labels, output_path):
    precision = [report[str(label)]["precision"] for label in labels]
    recall = [report[str(label)]["recall"] for label in labels]
    f1_scores = [report[str(label)]["f1-score"] for label in labels]

    x_positions = np.arange(len(labels))
    bar_width = 0.26

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(x_positions - bar_width, precision, bar_width, label="Precision")
    axis.bar(x_positions, recall, bar_width, label="Recall")
    axis.bar(x_positions + bar_width, f1_scores, bar_width, label="F1-score")

    axis.set(
        title="Per-Class Precision, Recall, and F1-score",
        xlabel="Digit",
        ylabel="Score",
        xticks=x_positions,
        xticklabels=labels,
        ylim=(0, 1.05),
    )
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_count_bar_chart(counts, labels, output_path, title, ylabel):
    values = [counts.get(label, 0) for label in labels]

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, values, color="#2563eb")

    axis.set(title=title, xlabel="Digit", ylabel=ylabel)
    axis.grid(axis="y", alpha=0.25)

    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_error_rate_chart(matrix, labels, output_path):
    totals = matrix.sum(axis=1)
    correct = np.diag(matrix)
    error_rates = np.divide(
        totals - correct,
        totals,
        out=np.zeros_like(totals, dtype=float),
        where=totals != 0,
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, error_rates * 100, color="#f43f5e")

    axis.set(
        title="Error Rate by Digit",
        xlabel="Digit",
        ylabel="Error rate (%)",
        ylim=(0, max(5, float(np.max(error_rates * 100)) * 1.2)),
    )
    axis.grid(axis="y", alpha=0.25)

    for bar, error_rate in zip(bars, error_rates):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{error_rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_misclassified_samples_chart(X_test, y_test, predictions, output_path, limit=24):
    missed_indices = np.where(y_test != predictions)[0][:limit]

    if len(missed_indices) == 0:
        return False

    columns = 6
    rows = int(np.ceil(len(missed_indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2.2))
    axes = np.atleast_1d(axes).flatten()

    for axis in axes:
        axis.axis("off")

    for axis, sample_index in zip(axes, missed_indices):
        image = X_test[sample_index].reshape(IMAGE_SIZE, IMAGE_SIZE)
        axis.imshow(image, cmap="gray")
        axis.set_title(
            f"Actual {y_test[sample_index]}\nPred {predictions[sample_index]}",
            fontsize=9,
        )

    figure.suptitle("Sample Misclassifications", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return True


def get_top_confusions(matrix, labels, limit=10):
    confusions = []

    for row_index, actual_label in enumerate(labels):
        for column_index, predicted_label in enumerate(labels):
            if row_index == column_index:
                continue

            count = int(matrix[row_index, column_index])
            if count > 0:
                confusions.append(
                    {
                        "actual": int(actual_label),
                        "predicted": int(predicted_label),
                        "count": count,
                    }
                )

    return sorted(confusions, key=lambda item: item["count"], reverse=True)[:limit]


def save_visualizations(model_id, config, X_test, y_test, predictions, accuracy, report, matrix):
    labels = list(range(10))
    output_dir = VISUALIZATION_DIR / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix_chart(
        matrix,
        labels,
        output_dir / "confusion_matrix.png",
        f"{config['display_name']} - Confusion Matrix",
    )
    save_confusion_matrix_chart(
        matrix,
        labels,
        output_dir / "confusion_matrix_normalized.png",
        f"{config['display_name']} - Normalized Confusion Matrix",
        normalize=True,
    )
    save_per_class_metrics_chart(
        report,
        labels,
        output_dir / "precision_recall_f1.png",
    )

    actual_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(y_test, return_counts=True))
    }
    prediction_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(predictions, return_counts=True))
    }

    save_count_bar_chart(
        actual_counts,
        labels,
        output_dir / "actual_digit_distribution.png",
        "Actual Digit Distribution in Test Set",
        "Test samples",
    )
    save_count_bar_chart(
        prediction_counts,
        labels,
        output_dir / "predicted_digit_distribution.png",
        "Predicted Digit Distribution",
        "Predictions",
    )
    save_error_rate_chart(matrix, labels, output_dir / "error_rate_by_digit.png")
    save_misclassified_samples_chart(
        X_test,
        y_test,
        predictions,
        output_dir / "misclassified_samples.png",
    )

    top_confusions = get_top_confusions(matrix, labels)
    summary = {
        "model_id": model_id,
        "display_name": config["display_name"],
        "dataset": config["dataset"],
        "accuracy": round(float(accuracy * 100), 2),
        "test_samples": int(len(y_test)),
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "top_confusions": top_confusions,
    }

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with (output_dir / "per_class_metrics.csv").open("w", encoding="utf-8") as file:
        file.write("digit,precision,recall,f1_score,support\n")
        for label in labels:
            metrics = report[str(label)]
            file.write(
                f"{label},"
                f"{metrics['precision']:.6f},"
                f"{metrics['recall']:.6f},"
                f"{metrics['f1-score']:.6f},"
                f"{int(metrics['support'])}\n"
            )

    with (output_dir / "top_confusions.csv").open("w", encoding="utf-8") as file:
        file.write("actual,predicted,count\n")
        for item in top_confusions:
            file.write(f"{item['actual']},{item['predicted']},{item['count']}\n")

    with (output_dir / "README.md").open("w", encoding="utf-8") as file:
        file.write(f"# {config['display_name']} Visualizations\n\n")
        file.write(f"- Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"- Test samples: {len(y_test)}\n")
        file.write("- Dataset: " + config["dataset"] + "\n\n")
        file.write("## Files\n\n")
        for file_name in [
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "precision_recall_f1.png",
            "actual_digit_distribution.png",
            "predicted_digit_distribution.png",
            "error_rate_by_digit.png",
            "misclassified_samples.png",
            "metrics_summary.json",
            "per_class_metrics.csv",
            "top_confusions.csv",
        ]:
            if (output_dir / file_name).exists():
                file.write(f"- `{file_name}`\n")

    print(f"\nSaved visualizations at: {output_dir}")


def train_and_save(model_id, X_train, X_test, y_train, y_test):
    config = MODEL_CONFIGS[model_id]

    print(f"\nTraining {config['display_name']}...")
    model = create_model()
    model.fit(X_train, y_train)

    print(f"Testing {config['display_name']}...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions, labels=list(range(10)))

    print("\nModel Accuracy:")
    print(f"{accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(matrix)

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
    save_visualizations(
        model_id,
        config,
        X_test,
        y_test,
        predictions,
        accuracy,
        report,
        matrix,
    )


def build_datasets(requested_model):
    datasets = {}

    needs_mnist = requested_model in {"all", "mnist", "combined"}
    needs_kaggle = requested_model in {"all", "kaggle", "combined"}
    needs_custom_combined = requested_model in {"all", "custom_combined"}

    if needs_mnist:
        try:
            datasets["mnist"] = load_mnist_data()
        except Exception as exc:
            print(f"Skipping MNIST model: {exc}")

    if needs_kaggle:
        try:
            datasets["kaggle"] = load_kaggle_data()
        except Exception as exc:
            print(f"Skipping Kaggle model: {exc}")

    if needs_custom_combined:
        try:
            datasets["custom_combined"] = load_custom_combined_data()
        except Exception as exc:
            print(f"Skipping custom combined model: {exc}")

    if requested_model in {"all", "combined"} and "mnist" in datasets and "kaggle" in datasets:
        print("Building combined MNIST + Kaggle dataset...")
        mnist_train, mnist_test, mnist_y_train, mnist_y_test = datasets["mnist"]
        kaggle_train, kaggle_test, kaggle_y_train, kaggle_y_test = datasets["kaggle"]

        datasets["combined"] = (
            np.vstack([mnist_train, kaggle_train]),
            np.vstack([mnist_test, kaggle_test]),
            np.concatenate([mnist_y_train, kaggle_y_train]),
            np.concatenate([mnist_y_test, kaggle_y_test]),
        )
    elif requested_model in {"all", "combined"}:
        print("Skipping combined model because one of the datasets is unavailable.")

    return datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train handwritten digit recognition models."
    )
    parser.add_argument(
        "--model",
        choices=["all", "mnist", "kaggle", "combined", "custom_combined"],
        default="all",
        help="Choose which model to train.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    datasets = build_datasets(args.model)

    if args.model == "all":
        model_ids = ["mnist", "kaggle", "combined", "custom_combined"]
    else:
        model_ids = [args.model]

    for model_id in model_ids:
        if model_id not in datasets:
            print(f"\nCannot train {model_id}: dataset was not prepared.")
            continue

        train_and_save(model_id, *datasets[model_id])


if __name__ == "__main__":
    main()
