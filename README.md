# Handwritten Digit Recognition System

A Flask-based handwritten digit recognition web app. Users can predict digits in two ways:

- Upload, drag, or paste a digit image.
- Draw a digit directly inside the app using the pencil canvas.

The system supports four selectable trained models:

- Model 1 - MNIST ubyte data
- Model 2 - Kaggle-style folder image dataset
- Model 3 - Combined MNIST + Kaggle dataset
- Model 4 - Handwritten folder + custom feedback dataset

## Project Structure

```text
HANDWRITTENDIGITSYSTEM/
|-- app.py
|-- requirements.txt
|-- Procfile
|-- start.py
|-- README.md
|-- data/
|   |-- mnist/
|   |   |-- train-images.idx3-ubyte
|   |   |-- train-labels.idx1-ubyte
|   |   |-- t10k-images.idx3-ubyte
|   |   `-- t10k-labels.idx1-ubyte
|   |-- handwritten_digits/
|   |   |-- 0/
|   |   |-- 1/
|   |   |-- ...
|   |   `-- 9/
|   `-- customize_data/
|       |-- 0/
|       |-- 1/
|       |-- ...
|       `-- 9/
|-- ml/
|   |-- preprocess.py
|   |-- train_model.py
|   |-- digit_model_mnist.joblib
|   |-- digit_model_kaggle.joblib
|   |-- digit_model_combined.joblib
|   `-- digit_model_custom_combined.joblib
|-- static/
|   |-- css/style.css
|   `-- js/app.js
`-- templates/
    `-- index.html
```

## Requirements

- Python 3.10 or newer
- Flask
- NumPy
- OpenCV
- scikit-learn
- joblib

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using the included virtual environment on Windows:

```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

### MNIST ubyte dataset

Place the MNIST ubyte files in:

```text
data/mnist/
```

Required files:

```text
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

### Kaggle folder image dataset

Place the image dataset in:

```text
data/handwritten_digits/
```

Expected layout:

```text
data/handwritten_digits/
|-- 0/
|-- 1/
|-- 2/
|-- 3/
|-- 4/
|-- 5/
|-- 6/
|-- 7/
|-- 8/
`-- 9/
```

Each folder should contain images of that digit. Supported formats are:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

## Training Models

Train all available models:

```bash
python ml/train_model.py
```

Train only one model:

```bash
python ml/train_model.py --model mnist
python ml/train_model.py --model kaggle
python ml/train_model.py --model combined
python ml/train_model.py --model custom_combined
```

The combined model is trained only when both MNIST and Kaggle datasets are available.
The custom combined model uses only `data/handwritten_digits` and `data/customize_data`.

Trained model files are saved in `ml/`:

```text
digit_model_mnist.joblib
digit_model_kaggle.joblib
digit_model_combined.joblib
digit_model_custom_combined.joblib
```

Current training results:

| Model | Dataset | Accuracy |
| --- | --- | ---: |
| Model 1 | MNIST ubyte | 97.22% |
| Model 2 | Kaggle folder images | 95.08% |
| Model 3 | Combined MNIST + Kaggle | 96.32% |
| Model 4 | Handwritten + custom feedback | 94.20% |

## Running the App

Start the Flask server:

```bash
python app.py
```

Or:

```bash
flask run
```

Open the app in your browser:

```text
http://127.0.0.1:5000/
```

## How to Use

1. Select a prediction model from the dropdown.
2. Choose an input mode:
   - Upload: browse, drag, or paste an image.
   - Draw: write a digit using the pencil canvas.
3. Click the prediction button.
4. The system displays:
   - Predicted digit
   - Confidence score
   - Selected model
   - Model accuracy
   - Input source

## Model Notes

The app uses `ExtraTreesClassifier` for the trained model bundles. This provides stable class probabilities through `predict_proba()`, which are used as the confidence score.

Images are preprocessed into the MNIST-style format:

- Grayscale
- White digit on black background
- Centered crop
- 28 x 28 pixels
- Flattened into 784 numeric features
- Pixel values normalized from `0` to `1`

## Useful Files

- [app.py](app.py): Flask routes, model loading, prediction logic
- [ml/train_model.py](ml/train_model.py): model training script
- [ml/preprocess.py](ml/preprocess.py): image preprocessing functions
- [templates/index.html](templates/index.html): app interface
- [static/js/app.js](static/js/app.js): upload, paste, drag/drop, and drawing behavior
- [static/css/style.css](static/css/style.css): app styling and animations
