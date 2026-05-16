import os
import subprocess
import sys


port = os.environ.get("PORT", "8080")

subprocess.run(
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
    check=True,
)
