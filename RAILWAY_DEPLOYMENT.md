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
Dockerfile
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
web: gunicorn --bind 0.0.0.0:$PORT --timeout 180 app:app
```

Railway provides the `PORT` environment variable automatically.

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
gunicorn --bind 0.0.0.0:$PORT --timeout 180 app:app
```

If Railway asks for a start command manually, use:

```text
gunicorn --bind 0.0.0.0:$PORT --timeout 180 app:app
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
- If OpenCV fails during deployment, keep `opencv-python-headless` in `requirements.txt`.
- If requests are too large, use smaller uploaded images or compress images before submitting feedback.
