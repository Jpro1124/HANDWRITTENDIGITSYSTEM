from pathlib import Path
import base64
from datetime import datetime
import json
import uuid
import warnings

import joblib
import numpy as np
from flask import Flask, render_template, request

from ml.preprocess import preprocess_image_from_upload


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml"
LEGACY_MODEL_PATH = MODEL_DIR / "digit_model.joblib"
CUSTOMIZE_DATA_DIR = BASE_DIR / "data" / "customize_data"
CUSTOMIZE_LOG_PATH = CUSTOMIZE_DATA_DIR / "feedback_log.jsonl"

MODEL_CONFIGS = {
    "mnist": {
        "display_name": "Model 1 - MNIST ubyte",
        "path": MODEL_DIR / "digit_model_mnist.joblib",
    },
    "kaggle": {
        "display_name": "Model 2 - Kaggle folder images",
        "path": MODEL_DIR / "digit_model_kaggle.joblib",
    },
    "combined": {
        "display_name": "Model 3 - Combined MNIST + Kaggle",
        "path": MODEL_DIR / "digit_model_combined.joblib",
    },
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
IMAGE_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/bmp": ".bmp",
}
EXTENSION_IMAGE_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "bmp": "image/bmp",
}

app = Flask(__name__)

model_bundles = {}


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_model_bundles():
    loaded_models = {}

    for model_id, config in MODEL_CONFIGS.items():
        if not config["path"].exists():
            continue

        bundle = joblib.load(config["path"])
        loaded_models[model_id] = {
            "id": model_id,
            "display_name": bundle.get("display_name", config["display_name"]),
            "dataset": bundle.get("dataset", "Unknown dataset"),
            "accuracy": bundle.get("accuracy"),
            "test_samples": bundle.get("test_samples"),
            "train_samples": bundle.get("train_samples"),
            "trained_at": bundle.get("trained_at"),
            "model": bundle["model"],
        }

    if not loaded_models and LEGACY_MODEL_PATH.exists():
        bundle = joblib.load(LEGACY_MODEL_PATH)
        loaded_models["mnist"] = {
            "id": "mnist",
            "display_name": "Model 1 - MNIST ubyte",
            "dataset": bundle.get("dataset", "MNIST ubyte data"),
            "accuracy": bundle.get("accuracy"),
            "test_samples": bundle.get("test_samples"),
            "train_samples": bundle.get("train_samples"),
            "trained_at": bundle.get("trained_at"),
            "model": bundle["model"],
        }

    if not loaded_models:
        raise FileNotFoundError(
            "Model files not found. Please train the models first by running: "
            "python ml/train_model.py"
        )

    return loaded_models


def get_available_models():
    return [
        {
            "id": bundle["id"],
            "display_name": bundle["display_name"],
            "dataset": bundle["dataset"],
            "accuracy": bundle["accuracy"],
            "test_samples": bundle["test_samples"],
            "train_samples": bundle["train_samples"],
            "trained_at": bundle["trained_at"],
        }
        for bundle in model_bundles.values()
    ]


def get_selected_model(model_id):
    if model_id in model_bundles:
        return model_bundles[model_id]

    return next(iter(model_bundles.values()))


def render_index(**kwargs):
    selected_model_id = kwargs.pop("selected_model_id", None)

    if selected_model_id is None:
        selected_model_id = next(iter(model_bundles.keys()))

    return render_template(
        "index.html",
        models=get_available_models(),
        selected_model_id=selected_model_id,
        **kwargs
    )


def predict_digit(file_bytes, selected_model):
    features = preprocess_image_from_upload(file_bytes)
    features = features.reshape(1, -1)

    model = selected_model["model"]
    prediction = model.predict(features)[0]
    confidence = calculate_confidence(model, features)

    return int(prediction), round(confidence, 2)


def image_data_url(file_bytes, mime_type):
    if mime_type not in IMAGE_MIME_EXTENSIONS:
        mime_type = "image/png"

    encoded_image = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def get_upload_mime_type(file):
    if file.mimetype in IMAGE_MIME_EXTENSIONS:
        return file.mimetype

    extension = file.filename.rsplit(".", 1)[1].lower()
    return EXTENSION_IMAGE_MIME_TYPES.get(extension, "image/png")


def calculate_confidence(model, features):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        probabilities = model.predict_proba(features)[0]

    if np.all(np.isfinite(probabilities)) and np.sum(probabilities) > 0:
        probabilities = probabilities / np.sum(probabilities)
        return float(np.max(probabilities) * 100)

    scores = model.decision_function(features)[0]
    shifted_scores = scores - np.max(scores)
    exp_scores = np.exp(shifted_scores)
    softmax_probabilities = exp_scores / np.sum(exp_scores)

    return float(np.max(softmax_probabilities) * 100)


def decode_canvas_image(data_url):
    if not data_url or "," not in data_url:
        raise ValueError("Please draw a digit before predicting.")

    header, encoded_image = data_url.split(",", 1)

    if "base64" not in header:
        raise ValueError("Invalid drawing data.")

    return base64.b64decode(encoded_image)


def parse_image_data_url(data_url):
    if not data_url or "," not in data_url:
        raise ValueError("Missing input image for feedback.")

    header, encoded_image = data_url.split(",", 1)
    header_parts = header.split(";")

    if len(header_parts) < 2 or header_parts[0].startswith("data:image/") is False:
        raise ValueError("Invalid feedback image data.")

    if "base64" not in header_parts:
        raise ValueError("Feedback image must be base64 encoded.")

    mime_type = header_parts[0].replace("data:", "", 1)
    if mime_type not in IMAGE_MIME_EXTENSIONS:
        raise ValueError("Unsupported feedback image type.")

    return mime_type, base64.b64decode(encoded_image)


def save_feedback_sample(image_data, source, predicted_digit, correct_digit, model_id):
    if str(predicted_digit) not in {str(number) for number in range(10)}:
        raise ValueError("Invalid predicted digit in feedback.")

    if correct_digit not in {str(number) for number in range(10)}:
        raise ValueError("Please choose the correct digit from 0 to 9.")

    mime_type, file_bytes = parse_image_data_url(image_data)
    extension = IMAGE_MIME_EXTENSIONS[mime_type]
    label_dir = CUSTOMIZE_DATA_DIR / correct_digit
    label_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_source = "".join(
        character for character in source.lower() if character.isalnum()
    ) or "sample"
    file_name = (
        f"{timestamp}_{uuid.uuid4().hex[:8]}_"
        f"pred-{predicted_digit}_actual-{correct_digit}_{safe_source}{extension}"
    )
    sample_path = label_dir / file_name
    sample_path.write_bytes(file_bytes)

    log_entry = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "file": str(sample_path.relative_to(BASE_DIR)),
        "source": source,
        "predicted_digit": int(predicted_digit),
        "correct_digit": int(correct_digit),
        "model_id": model_id,
    }

    CUSTOMIZE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CUSTOMIZE_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    return sample_path


@app.route("/")
def index():
    return render_index()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_mode = request.form.get("input_mode", "upload")
        selected_model = get_selected_model(request.form.get("model_id", ""))

        if input_mode == "draw":
            file_bytes = decode_canvas_image(request.form.get("draw_image"))
            input_image_data_url = image_data_url(file_bytes, "image/png")
            prediction, confidence = predict_digit(file_bytes, selected_model)
            source = "Drawing"
        else:
            if "digit_image" not in request.files:
                return render_index(
                    error="No image file uploaded.",
                    active_mode="upload",
                    selected_model_id=selected_model["id"]
                )

            file = request.files["digit_image"]

            if file.filename == "":
                return render_index(
                    error="Please select, drag, or paste an image first.",
                    active_mode="upload",
                    selected_model_id=selected_model["id"]
                )

            if not allowed_file(file.filename):
                return render_index(
                    error="Invalid file type. Please upload PNG, JPG, JPEG, or BMP.",
                    active_mode="upload",
                    selected_model_id=selected_model["id"]
                )

            file_bytes = file.read()
            input_image_data_url = image_data_url(file_bytes, get_upload_mime_type(file))
            prediction, confidence = predict_digit(file_bytes, selected_model)
            source = "Uploaded image"

        return render_index(
            prediction=int(prediction),
            confidence=confidence,
            source=source,
            active_mode=input_mode,
            input_image_data_url=input_image_data_url,
            input_mode=input_mode,
            selected_model=selected_model,
            selected_model_id=selected_model["id"]
        )

    except Exception as e:
        return render_index(
            error=f"Prediction failed: {str(e)}",
            active_mode=request.form.get("input_mode", "upload"),
            selected_model_id=request.form.get("model_id")
        )


@app.route("/feedback", methods=["POST"])
def feedback():
    selected_model = get_selected_model(request.form.get("model_id", ""))
    input_mode = request.form.get("input_mode", "upload")
    source = request.form.get("source", "Input image")
    predicted_digit = request.form.get("predicted_digit", "")
    correct_digit = request.form.get("correct_digit", "")
    confidence = request.form.get("confidence")
    input_image_data_url = request.form.get("input_image_data_url")

    try:
        save_feedback_sample(
            image_data=input_image_data_url,
            source=source,
            predicted_digit=predicted_digit,
            correct_digit=correct_digit,
            model_id=selected_model["id"],
        )
        if str(predicted_digit) == str(correct_digit):
            feedback_context = "confirmed correct"
        else:
            feedback_context = "corrected"

        feedback_message = (
            f"Saved this {feedback_context} sample as digit {correct_digit} in "
            "data/customize_data for retraining."
        )
        feedback_error = None
    except Exception as exc:
        feedback_message = None
        feedback_error = f"Could not save feedback: {exc}"

    return render_index(
        prediction=int(predicted_digit) if str(predicted_digit).isdigit() else None,
        confidence=confidence,
        source=source,
        active_mode=input_mode,
        input_mode=input_mode,
        input_image_data_url=input_image_data_url,
        selected_model=selected_model,
        selected_model_id=selected_model["id"],
        feedback_message=feedback_message,
        feedback_error=feedback_error,
        correct_digit=correct_digit,
    )


model_bundles = load_model_bundles()


if __name__ == "__main__":
    app.run(debug=True)
