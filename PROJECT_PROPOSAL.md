# ITBAN 5 Final Project Proposal

## 1. Project Overview

### 1.1. Title

Handwritten Digit Recognition System Using Machine Learning

### 1.2. Description

This project aims to develop a web-based handwritten digit recognition system that can classify digits from `0` to `9`. Users can either upload an image of a handwritten digit or draw a digit directly inside the system using a pencil canvas feature.

The problem addressed by this project is the need for automatic recognition of handwritten numeric characters. Handwritten digit recognition is an important machine learning task because it is used in real-world applications such as form processing, bank check recognition, postal code reading, and document digitization.

The system uses multiple trained models so users can compare predictions from different datasets:

- Model 1: MNIST ubyte dataset
- Model 2: Kaggle-style handwritten digit image folders
- Model 3: Combined MNIST and Kaggle dataset

### 1.3. Objectives

Primary Objectives:

- Build a handwritten digit recognition system that predicts digits from `0` to `9`.
- Allow users to choose between multiple trained models.
- Provide two prediction input methods: image upload and in-system drawing.
- Display the predicted digit, confidence score, selected model, and model accuracy.
- Train and evaluate models using MNIST, Kaggle folder images, and a combined dataset.

Secondary Objectives:

- Create an interactive and user-friendly web interface.
- Improve usability with drag-and-drop, paste upload, and pencil drawing support.
- Document the training process, dataset structure, and deployment steps.
- Compare model performance using accuracy and classification metrics.

## 2. Data Collection and Preparation

### 2.1. Data Source

Source:

- MNIST ubyte dataset
- Kaggle-style handwritten digit image dataset
- Combined dataset created from both MNIST and Kaggle data

Description:

The MNIST dataset contains grayscale images of handwritten digits from `0` to `9`. The files are stored in ubyte format and include separate training and testing image/label files.

The Kaggle-style dataset is organized into folders named `0` through `9`. Each folder contains handwritten digit image files for the corresponding label. In this project, the dataset contains `.jpg` images.

Dataset layout:

```text
data/
|-- mnist/
|   |-- train-images.idx3-ubyte
|   |-- train-labels.idx1-ubyte
|   |-- t10k-images.idx3-ubyte
|   `-- t10k-labels.idx1-ubyte
`-- handwritten_digits/
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

### 2.2. Data Preprocessing

Steps Involved:

- Convert uploaded or dataset images to grayscale.
- Resize large images for faster processing.
- Apply Gaussian blur to reduce noise.
- Detect whether the background is light or dark.
- Apply thresholding so the digit becomes white on a black background.
- Find the digit contour and crop the digit region.
- Center the digit in a square image.
- Add padding around the digit.
- Resize the final image to `28 x 28` pixels.
- Normalize pixel values from `0-255` to `0-1`.
- Flatten each image into `784` features.

Tools/Techniques Used:

- OpenCV for image reading, thresholding, contour detection, cropping, resizing, and preprocessing.
- NumPy for array operations and normalization.
- scikit-learn utilities for dataset splitting and model training.

## 3. Model Selection and Training

### 3.1. Model Architecture

Chosen Model:

The project uses `ExtraTreesClassifier`, an ensemble-based machine learning model from scikit-learn.

Justification:

`ExtraTreesClassifier` was selected because it performs well on structured feature data such as flattened `28 x 28` grayscale images. It also provides stable class probability outputs through `predict_proba()`, which allows the system to display a confidence score for each prediction.

Compared with the previous SGD-based classifier, `ExtraTreesClassifier` produced stable confidence values and avoided invalid probability outputs. It is also practical for this project because it trains directly on flattened image features without requiring a deep learning pipeline.

### 3.2. Training Process

Training Data:

- Model 1 uses the MNIST training set with `60,000` training images.
- Model 2 uses `80%` of the Kaggle-style folder image dataset for training.
- Model 3 combines the MNIST training set and the Kaggle training split.

Validation/Test Data:

- Model 1 uses the MNIST test set with `10,000` test images.
- Model 2 uses `20%` of the Kaggle-style folder image dataset as the test set.
- Model 3 combines the MNIST test set and the Kaggle test split.

Hyperparameters:

```text
Model: ExtraTreesClassifier
n_estimators: 160
random_state: 42
n_jobs: -1
class_weight: balanced
Kaggle test_size: 0.2
Kaggle split: stratified by label
Image size: 28 x 28
Input features: 784
```

### 3.3. Tools and Libraries

Frameworks:

- Flask
- scikit-learn
- OpenCV
- NumPy
- joblib

Development Environment:

- Python virtual environment
- Visual Studio Code
- Windows local development environment
- Flask local server

## 4. Model Evaluation

### 4.1. Evaluation Metrics

Metrics Used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Confidence score during prediction

Justification:

Accuracy is useful because the digit classes are clearly defined from `0` to `9`. Precision, recall, and F1-score provide more detailed insight into how well each digit class is predicted. The confusion matrix helps identify which digits are commonly misclassified. Confidence score is useful in the deployed system because it shows how certain the selected model is about its prediction.

### 4.2. Results

Performance:

| Model | Dataset | Test Samples | Accuracy |
| --- | --- | ---: | ---: |
| Model 1 | MNIST ubyte | 10,000 | 97.22% |
| Model 2 | Kaggle folder images | 4,311 | 95.08% |
| Model 3 | Combined MNIST + Kaggle | 14,311 | 96.32% |

Analysis:

Model 1 achieved the highest accuracy on the MNIST test set with `97.22%`. This is expected because the MNIST dataset has clean and standardized digit images.

Model 2 achieved `95.08%` accuracy on the Kaggle-style folder dataset. This result shows that the model can learn from custom image folders, although real image variation may make the task more difficult than MNIST.

Model 3 achieved `96.32%` accuracy using the combined dataset. This model is useful because it is trained on both standardized MNIST data and additional handwritten digit images, giving it broader exposure to different handwriting styles.

Screenshots of model results and the deployed prediction interface can be added in the final documentation after testing the application in the browser.

## 5. Deployment

### 5.1. Deployment Strategy

Platform:

The project is deployed as a local web application using Flask. Users access the system through a browser at:

```text
http://127.0.0.1:5000/
```

Tools/Frameworks:

- Flask for the web application backend.
- HTML, CSS, and JavaScript for the user interface.
- OpenCV and NumPy for image preprocessing.
- scikit-learn and joblib for loading trained machine learning models.

Deployment Features:

- Model selector for choosing between MNIST, Kaggle, and combined models.
- Upload, drag, and paste image prediction.
- Pencil canvas for drawing a digit directly inside the system.
- Prediction result display with digit, confidence score, model name, model accuracy, and input source.

