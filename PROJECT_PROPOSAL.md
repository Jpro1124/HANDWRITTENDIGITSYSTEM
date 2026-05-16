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
- Model 4: Kaggle-style handwritten digit image folders combined with custom feedback data collected from users

### 1.3. Objectives

Primary Objectives:

- Build a handwritten digit recognition system that predicts digits from `0` to `9`.
- Allow users to choose between multiple trained models.
- Provide two prediction input methods: image upload and in-system drawing.
- Display the input image or drawing beside the predicted result for live comparison.
- Display the predicted digit, confidence score, selected model, and model accuracy.
- Collect user feedback when predictions are correct or incorrect.
- Save confirmed and corrected samples into a custom dataset for retraining.
- Train and evaluate models using MNIST, Kaggle folder images, combined data, and custom feedback data.

Secondary Objectives:

- Create an interactive and user-friendly web interface.
- Improve usability with drag-and-drop, paste upload, and pencil drawing support.
- Document the training process, dataset structure, and deployment steps.
- Compare model performance using accuracy, precision, recall, F1-score, confusion matrix, and generated visualization reports.

## 2. Data Collection and Preparation

### 2.1. Data Source

Source:

- MNIST ubyte dataset
- Kaggle-style handwritten digit image dataset
- Combined dataset created from both MNIST and Kaggle data
- Custom feedback dataset collected from user confirmations and corrections

Description:

The MNIST dataset contains grayscale images of handwritten digits from `0` to `9`. The files are stored in ubyte format and include separate training and testing image/label files.

The Kaggle-style dataset is organized into folders named `0` through `9`. Each folder contains handwritten digit image files for the corresponding label. In this project, the dataset contains `.jpg` images.

The custom feedback dataset is created by the application. After a prediction, users can confirm whether the output is correct. If the output is wrong, users select the correct digit. The system saves both confirmed correct samples and corrected samples into `data/customize_data/`, organized by digit label. This dataset is used to improve future training.

Dataset layout:

```text
data/
|-- mnist/
|   |-- train-images.idx3-ubyte
|   |-- train-labels.idx1-ubyte
|   |-- t10k-images.idx3-ubyte
|   `-- t10k-labels.idx1-ubyte
|-- handwritten_digits/
|   |-- 0/
|   |-- 1/
|   |-- ...
|   `-- 9/
`-- customize_data/
    |-- 0/
    |-- 1/
    |-- ...
    |-- 9/
    `-- feedback_log.jsonl
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
- Save feedback images under the correct digit folder when users confirm or correct a prediction.

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
- Model 4 combines the Kaggle-style handwritten digit folder dataset and the custom feedback dataset only. MNIST is excluded from Model 4 to avoid relying on digit styles that were less accurate for some handwritten `4`, `7`, and `9` samples.

Validation/Test Data:

- Model 1 uses the MNIST test set with `10,000` test images.
- Model 2 uses `20%` of the Kaggle-style folder image dataset as the test set.
- Model 3 combines the MNIST test set and the Kaggle test split.
- Model 4 uses a stratified `20%` split from the handwritten folder and custom feedback data.

Hyperparameters:

```text
Model: ExtraTreesClassifier
n_estimators: 160
random_state: 42
n_jobs: -1
class_weight: balanced
Kaggle test_size: 0.2
Kaggle split: stratified by label
Custom combined split: stratified by label
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
- Matplotlib

Development Environment:

- Python virtual environment
- Visual Studio Code
- Windows local development environment
- Flask local server
- GitHub repository
- Railway deployment environment

## 4. Model Evaluation

### 4.1. Evaluation Metrics

Metrics Used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Normalized confusion matrix
- Error rate per digit
- Per-class metric visualization
- Actual and predicted digit distribution charts
- Misclassified sample visualization
- Confidence score during prediction

Justification:

Accuracy is useful because the digit classes are clearly defined from `0` to `9`. Precision, recall, and F1-score provide more detailed insight into how well each digit class is predicted. The confusion matrix and normalized confusion matrix help identify which digits are commonly misclassified. Error-rate and distribution charts show whether the model struggles with specific labels or predicts some digits too often. Confidence score is useful in the deployed system because it shows how certain the selected model is about its prediction.

### 4.2. Results

Performance:

| Model | Dataset | Test Samples | Accuracy |
| --- | --- | ---: | ---: |
| Model 1 | MNIST ubyte | 10,000 | 97.22% |
| Model 2 | Kaggle folder images | 4,311 | 95.08% |
| Model 3 | Combined MNIST + Kaggle | 14,311 | 96.32% |
| Model 4 | Handwritten folder + custom feedback | 4,538 | 94.20% |

Analysis:

Model 1 achieved the highest accuracy on the MNIST test set with `97.22%`. This is expected because the MNIST dataset has clean and standardized digit images.

Model 2 achieved `95.08%` accuracy on the Kaggle-style folder dataset. This result shows that the model can learn from custom image folders, although real image variation may make the task more difficult than MNIST.

Model 3 achieved `96.32%` accuracy using the combined dataset. This model is useful because it is trained on both standardized MNIST data and additional handwritten digit images, giving it broader exposure to different handwriting styles.

Model 4 achieved `94.20%` accuracy using only the handwritten folder dataset and the custom feedback dataset. This model is important because it focuses on the dataset style collected by the system and avoids MNIST samples that may not match some user-written digits, especially difficult classes such as `4`, `7`, and `9`.

The latest Model 4 macro average scores are:

| Metric | Score |
| --- | ---: |
| Precision | 94.22% |
| Recall | 94.18% |
| F1-score | 94.18% |

The generated visual reports are saved in:

```text
visualization/custom_combined/
```

Important generated files include:

- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `precision_recall_f1.png`
- `actual_digit_distribution.png`
- `predicted_digit_distribution.png`
- `error_rate_by_digit.png`
- `misclassified_samples.png`
- `metrics_summary.json`
- `per_class_metrics.csv`
- `top_confusions.csv`

The top confusion cases for Model 4 include `4` predicted as `1`, `5` predicted as `3`, and `5` predicted as `6`. These results can guide future collection of more training samples for the most confusing digit pairs.

## 5. Deployment

### 5.1. Deployment Strategy

Platform:

The project will be hosted using GitHub and Railway. GitHub will store the project source code, trained model files, templates, static assets, and deployment configuration. Railway will build and run the Flask web application from the GitHub repository using the included `Procfile`. The Dockerfile remains available as an optional alternative deployment method.

Tools/Frameworks:

- Flask for the web application backend.
- HTML, CSS, and JavaScript for the user interface.
- OpenCV and NumPy for image preprocessing.
- scikit-learn and joblib for loading trained machine learning models.
- GitHub for version control and repository hosting.
- Railway for cloud deployment and public web hosting.
- Gunicorn as the production WSGI server.
- Procfile for the Railway web start command.
- Dockerfile as an optional container deployment setup.

Deployment Process:

- Push the completed project files to a GitHub repository.
- Connect the GitHub repository to a Railway project.
- Railway installs the required Python packages from `requirements.txt`.
- Railway runs the Flask app using the `Procfile` command and the platform-provided `PORT`.
- Users access the deployed system through the public Railway application URL.

Deployment Features:

- Model selector for choosing between trained digit recognition models.
- Upload, drag, and paste image prediction.
- Pencil canvas for drawing a digit directly inside the system.
- Prediction result display with digit, confidence score, model name, model accuracy, and input source.
- Live comparison between the submitted image or drawing and the predicted result.
- Feedback saving for correct and corrected outputs, which can be used for future retraining.

## 6. Expected Output

The final system will provide a deployed web application where users can upload or draw a handwritten digit, select a trained model, and receive a prediction with confidence score. The system will also show the input image beside the prediction so users can compare the result immediately.

When users confirm or correct predictions, the system will save the image into the custom feedback dataset. These collected samples can be used to retrain Model 4 and improve recognition for handwriting styles that are not well represented in the original datasets.

The project also produces training evaluation artifacts in the `visualization/` folder. These visualizations help explain model performance through confusion matrices, precision/recall/F1 charts, digit distribution charts, error-rate charts, and misclassified sample examples.

## 7. Conclusion

This project demonstrates a complete machine learning workflow for handwritten digit recognition, including data preprocessing, model training, web deployment, user interaction, feedback collection, retraining support, and evaluation visualization. The system is designed to improve over time by collecting corrected and confirmed user samples, making it more adaptable to real handwriting inputs.

