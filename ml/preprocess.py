import cv2
import numpy as np


IMAGE_SIZE = 28


def preprocess_gray_image(gray_image):
    """
    Preprocesses a handwritten digit image using OpenCV.

    Output format:
    - White digit
    - Black background
    - 28x28 pixels
    - Flattened into 784 features
    - Pixel values from 0 to 1
    """

    if gray_image is None:
        raise ValueError("Invalid image. Could not read image file.")

    # Resize very large images first for faster processing
    height, width = gray_image.shape

    if max(height, width) > 1000:
        scale = 1000 / max(height, width)
        gray_image = cv2.resize(
            gray_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    # Blur to reduce small noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect background color
    # If background is mostly white, invert so digit becomes white.
    if np.mean(blurred) > 127:
        threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        threshold_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        threshold_type
    )

    # Find the digit area
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        if w > 0 and h > 0:
            digit_crop = thresh[y:y + h, x:x + w]
        else:
            digit_crop = thresh
    else:
        digit_crop = thresh

    # Make the image square
    h, w = digit_crop.shape
    square_size = max(h, w)

    square_image = np.zeros((square_size, square_size), dtype=np.uint8)

    x_offset = (square_size - w) // 2
    y_offset = (square_size - h) // 2

    square_image[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = digit_crop

    # Add padding around the digit
    padded_image = cv2.copyMakeBorder(
        square_image,
        10,
        10,
        10,
        10,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Resize to 28x28
    resized_image = cv2.resize(
        padded_image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Normalize from 0-255 to 0-1
    normalized_image = resized_image.astype("float32") / 255.0

    # Flatten to 784 features
    flattened_image = normalized_image.flatten()

    return flattened_image


def preprocess_image_from_path(image_path):
    gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return preprocess_gray_image(gray_image)


def preprocess_image_from_upload(file_bytes):
    file_array = np.frombuffer(file_bytes, np.uint8)

    gray_image = cv2.imdecode(
        file_array,
        cv2.IMREAD_GRAYSCALE
    )

    return preprocess_gray_image(gray_image)