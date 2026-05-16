# Railway Deployment Guide

This guide deploys the Handwritten Digit Recognition System using GitHub and Railway.

## 1. Deployment Choice

Railway can deploy this Flask app in two ways:

- **Procfile buildpack deployment**: Railway installs Python dependencies from `requirements.txt` and runs the command in `Procfile`.
- **Dockerfile deployment**: Railway builds the included Dockerfile and runs the container.

For this project, the recommended setup is the **Procfile buildpack deployment** because it is simpler for a Flask app. The Dockerfile can stay in the repository as an alternative deployment option.

## 2. Required Files

Make sure these files are committed:

```text
app.py
requirements.txt
Procfile
start.py
nixpacks.toml
README.md
.gitignore
.gitattributes
templates/
static/
ml/
visualization/
```

The `Procfile` should contain:

```text
web: python start.py
```

Railway provides the `PORT` environment variable automatically. The `start.py` launcher reads that value and starts Gunicorn with a valid numeric port.

The `nixpacks.toml` file installs Git LFS and runs `git lfs pull` during the Railway build. This is required because the `.joblib` model files are stored with Git LFS.

## 3. Model Files and Git LFS

The trained `.joblib` model files can be large, especially:

```text
ml/digit_model_custom_combined.joblib
```

Use Git LFS for model files:

```powershell
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Track model files with Git LFS"
```

If `.gitattributes` is already committed and includes `*.joblib`, you do not need to track it again.

Before pushing, confirm that Git LFS has uploaded the actual model files:

```powershell
git lfs ls-files
```

## 4. Push to GitHub

Stage and commit the project:

```powershell
git add .
git commit -m "Deploy digit recognition app to Railway"
```

Push to GitHub:

```powershell
git push origin main
```

If your branch is named `master`, use:

```powershell
git push origin master
```

## 5. Create Railway Project

1. Go to `https://railway.app/`.
2. Create a new project.
3. Choose **Deploy from GitHub repo**.
4. Select the repository for this project.
5. Railway will detect the Python app and use the `Procfile`.

## 6. Environment and Start Command

No custom environment variables are required for prediction.

Railway should run:

```text
python start.py
```

If Railway asks for a start command manually, use:

```text
python start.py
```

## 7. Verify Deployment

After Railway finishes building:

1. Open the generated Railway public URL.
2. Confirm the app loads.
3. Select a model.
4. Upload or draw a digit.
5. Check that the prediction result appears.

## 8. Notes

- Do not commit raw datasets unless needed. The app only needs trained `.joblib` models for prediction.
- If Railway build fails because of large files, confirm Git LFS is enabled and the model files are uploaded through LFS.
- If Railway logs show `KeyError: 118`, Railway is loading a Git LFS pointer instead of the real model file. Confirm `nixpacks.toml` is committed and that `git lfs pull` runs during build.
- If OpenCV fails during deployment, keep `opencv-python-headless` in `requirements.txt`.
- If requests are too large, use smaller uploaded images or compress images before submitting feedback.
