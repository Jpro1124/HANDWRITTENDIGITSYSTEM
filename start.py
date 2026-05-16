import os
from pathlib import Path
import subprocess
import sys


MODEL_DIR = Path(__file__).resolve().parent / "ml"


def is_git_lfs_pointer(file_path):
    try:
        with file_path.open("rb") as file:
            return file.read(128).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return False


def pull_lfs_models_if_needed():
    model_files = list(MODEL_DIR.glob("*.joblib"))

    if not any(is_git_lfs_pointer(model_file) for model_file in model_files):
        return

    if not (Path(__file__).resolve().parent / ".git").exists():
        raise RuntimeError(
            "Detected Git LFS pointer model files, but Railway's build image does "
            "not include the .git folder, so `git lfs pull` cannot run here. "
            "Upload the real model files through Git LFS and enable Railway/GitHub "
            "LFS checkout, or use a Docker deploy that performs a full git clone."
        )

    print("Detected Git LFS pointer model files. Running git lfs pull...", flush=True)
    subprocess.run(["git", "lfs", "install"], check=False)
    subprocess.run(["git", "lfs", "pull"], check=True)

    pointer_files = [
        model_file.name
        for model_file in model_files
        if is_git_lfs_pointer(model_file)
    ]

    if pointer_files:
        raise RuntimeError(
            "Git LFS pull finished, but these model files are still pointers: "
            + ", ".join(pointer_files)
        )


port = os.environ.get("PORT", "8080")
pull_lfs_models_if_needed()

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
