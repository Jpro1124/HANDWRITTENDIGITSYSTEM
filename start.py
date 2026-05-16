import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlretrieve


MODEL_DIR = Path(__file__).resolve().parent / "ml"
MODEL_URL_ENV_VARS = {
    "digit_model_mnist.joblib": "MODEL_MNIST_URL",
    "digit_model_kaggle.joblib": "MODEL_KAGGLE_URL",
    "digit_model_combined.joblib": "MODEL_COMBINED_URL",
    "digit_model_custom_combined.joblib": "MODEL_CUSTOM_COMBINED_URL",
}


def is_git_lfs_pointer(file_path):
    try:
        with file_path.open("rb") as file:
            return file.read(128).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return False


def download_model(model_file, url):
    temp_path = model_file.with_suffix(model_file.suffix + ".download")

    print(f"Downloading {model_file.name} from configured model URL...", flush=True)
    urlretrieve(url, temp_path)
    temp_path.replace(model_file)


def prepare_models():
    model_files = list(MODEL_DIR.glob("*.joblib"))

    if not any(is_git_lfs_pointer(model_file) for model_file in model_files):
        return

    for model_file in model_files:
        if not is_git_lfs_pointer(model_file):
            continue

        env_var = MODEL_URL_ENV_VARS.get(model_file.name)
        model_url = os.environ.get(env_var, "") if env_var else ""

        if model_url:
            download_model(model_file, model_url)

    if not any(is_git_lfs_pointer(model_file) for model_file in model_files):
        return

    if (Path(__file__).resolve().parent / ".git").exists():
        print("Detected Git LFS pointer model files. Running git lfs pull...", flush=True)
        subprocess.run(["git", "lfs", "install"], check=False)
        subprocess.run(["git", "lfs", "pull"], check=False)

    pointer_files = [
        model_file.name
        for model_file in model_files
        if is_git_lfs_pointer(model_file)
    ]

    if pointer_files:
        print(
            "These model files are still Git LFS pointers and will be skipped: "
            + ", ".join(pointer_files),
            flush=True,
        )


port = os.environ.get("PORT", "8080")
prepare_models()

os.execv(
    sys.executable,
    [
        sys.executable,
        "-m",
        "gunicorn",
        "--bind",
        f"0.0.0.0:{port}",
        "--timeout",
        "180",
        "app:app",
    ],
)
