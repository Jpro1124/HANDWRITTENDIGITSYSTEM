# Hugging Face Spaces Deployment Guide

This guide deploys the Handwritten Digit Recognition System as a Hugging Face Space using Docker.

## 1. What Hugging Face Spaces Needs

This project is a Flask app, so the best Hugging Face Spaces option is:

```text
SDK: Docker
Port: 7860
```

The included deployment files are:

```text
Dockerfile
.dockerignore
.gitattributes
.gitignore
README.md
requirements.txt
```

Important notes:

- Hugging Face Docker Spaces use the port configured in the README YAML block.
- This project uses `app_port: 7860`.
- The `Dockerfile` runs the app using Gunicorn on `0.0.0.0:7860`.
- The `.joblib` model files are large, so they should be tracked with Git LFS.

Official references:

- Docker Spaces: https://huggingface.co/docs/hub/main/en/spaces-sdks-docker
- Spaces overview: https://huggingface.co/docs/hub/main/spaces
- GitHub Actions / Git LFS note for Spaces: https://huggingface.co/docs/hub/main/spaces-github-actions

## 2. Prepare Your Local Project

Open PowerShell in the project folder:

```powershell
cd C:\Users\insgr\Desktop\HANDWRITTENDIGITSYSTEM
```

Check that these model files exist:

```powershell
Get-ChildItem ml\digit_model*.joblib
```

Expected model files:

```text
ml/digit_model_mnist.joblib
ml/digit_model_kaggle.joblib
ml/digit_model_combined.joblib
```

You do not need to upload the raw datasets to Hugging Face Spaces for prediction. The trained model files are enough.

## 3. Install Git LFS

The trained model files are larger than normal Git file limits. Use Git LFS.

Install Git LFS from:

```text
https://git-lfs.com/
```

Then run:

```powershell
git lfs install
git lfs track "*.joblib"
```

The project already includes:

```text
.gitattributes
```

with:

```text
*.joblib filter=lfs diff=lfs merge=lfs -text
```

## 4. Create a Hugging Face Space

1. Go to:

```text
https://huggingface.co/spaces
```

2. Click **Create new Space**.

3. Fill in:

```text
Space name: handwritten-digit-recognition
License: choose what your school/project requires
SDK: Docker
Visibility: Public or Private
```

4. Create the Space.

Hugging Face will give you a repository URL similar to:

```text
https://huggingface.co/spaces/YOUR_USERNAME/handwritten-digit-recognition
```

## 5. Connect Your Local Project to the Space

If your local project already has Git initialized, add the Space remote:

```powershell
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/handwritten-digit-recognition
```

If the remote already exists, update it:

```powershell
git remote set-url space https://huggingface.co/spaces/YOUR_USERNAME/handwritten-digit-recognition
```

Replace `YOUR_USERNAME` with your Hugging Face username.

## 6. Stage Only Deployment Files

Do not push the raw datasets or virtual environment. The `.gitignore` helps prevent that, but if files were already staged before, unstage them first:

```powershell
git reset
```

Then stage only the deployment files:

```powershell
git add .gitattributes .gitignore .dockerignore Dockerfile README.md HUGGINGFACE_DEPLOYMENT.md
git add app.py requirements.txt
git add templates static ml\preprocess.py ml\train_model.py ml\__init__.py
git add ml\digit_model_mnist.joblib ml\digit_model_kaggle.joblib ml\digit_model_combined.joblib
```

Check what will be committed:

```powershell
git status --short
```

Make sure these are not included:

```text
venv/
data/
__pycache__/
```

## 7. Commit and Push

Commit your deployment files:

```powershell
git commit -m "Deploy Flask digit recognition app to Hugging Face Spaces"
```

Push to the Space:

```powershell
git push space main
```

If your branch is named `master`, use:

```powershell
git push space master:main
```

## 8. Wait for the Space to Build

Go to your Space page:

```text
https://huggingface.co/spaces/YOUR_USERNAME/handwritten-digit-recognition
```

Open the **Logs** tab.

You should see Docker build steps such as:

```text
Installing requirements
Starting gunicorn
Listening at: http://0.0.0.0:7860
```

When the build finishes, the Space should show the web app.

## 9. Test the Deployed App

In the Hugging Face Space:

1. Select a model from the dropdown.
2. Upload, drag, or paste a digit image.
3. Try the drawing canvas.
4. Confirm that the prediction result shows:

```text
Predicted digit
Confidence
Model name
Model accuracy
Source
```

## 10. Common Problems

### Space is stuck on Starting

Check that:

- `README.md` has `sdk: docker`.
- `README.md` has `app_port: 7860`.
- `Dockerfile` exposes port `7860`.
- Gunicorn binds to `0.0.0.0:7860`.

### Model files are missing

Check that `.joblib` files were pushed with Git LFS:

```powershell
git lfs ls-files
```

### Push is too large

Make sure you did not commit:

```text
data/
venv/
__pycache__/
```

If those were staged, run:

```powershell
git reset
```

then repeat the staging commands in section 6.

### OpenCV import error

This deployment uses:

```text
opencv-python-headless
```

instead of:

```text
opencv-python
```

This is better for server deployments because it avoids unnecessary GUI dependencies.

## 11. Updating the App Later

After making changes:

```powershell
git add app.py templates static README.md
git commit -m "Update deployed app"
git push space main
```

If you retrain models:

```powershell
python ml\train_model.py
git add ml\digit_model_mnist.joblib ml\digit_model_kaggle.joblib ml\digit_model_combined.joblib
git commit -m "Update trained digit models"
git push space main
```

