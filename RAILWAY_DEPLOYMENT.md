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

## 3. Model Files

The trained `.joblib` model files can be large, especially:

```text
ml/digit_model_custom_combined.joblib
```

Railway may receive Git LFS pointer files instead of the real model files. The most reliable GitHub + Railway setup is to upload the real model file as a **GitHub Release asset**, then give Railway the download URL.

Recommended Railway model:

```text
ml/digit_model_custom_combined.joblib
```

### Option A: GitHub Release Asset

1. Open your GitHub repository.
2. Go to **Releases**.
3. Create a new release, for example `models-v1`.
4. Upload `ml/digit_model_custom_combined.joblib` as a release asset.
5. Copy the asset download URL.
6. In Railway, add this environment variable:

```text
MODEL_CUSTOM_COMBINED_URL=<your GitHub release asset download URL>
```

On startup, `start.py` downloads the model from this URL if the repository contains only a Git LFS pointer.

### Option B: Git LFS

Use Git LFS for model files if your deployment platform checks out LFS objects correctly:

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
git lfs push --all origin main
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

If Railway receives Git LFS pointer files, set at least one model URL environment variable:

```text
MODEL_CUSTOM_COMBINED_URL=<your GitHub release asset download URL>
```

Optional model URL variables:

```text
MODEL_MNIST_URL
MODEL_KAGGLE_URL
MODEL_COMBINED_URL
MODEL_CUSTOM_COMBINED_URL
```

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
- If Railway logs show `Git LFS pointer`, set `MODEL_CUSTOM_COMBINED_URL` to a real model download URL.
- If OpenCV fails during deployment, keep `opencv-python-headless` in `requirements.txt`.
- If requests are too large, use smaller uploaded images or compress images before submitting feedback.
