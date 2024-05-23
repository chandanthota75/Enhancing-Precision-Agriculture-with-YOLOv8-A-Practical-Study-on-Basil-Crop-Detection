## Enhancing Precision Agriculture with YOLOv8: A Practical Study on Basil Crop Detection

This project delves into the application of YOLOv8, a state-of-the-art object detection model, for real-world use cases in precision agriculture. Here, we focus on automating basil crop identification within agricultural fields. By leveraging a deep learning approach, this project aims to enhance efficiency and accuracy in crop monitoring tasks, empowering farmers with valuable insights.

This readme serves as a comprehensive guide to the project, outlining the methodology, implementation details, and key findings. We'll delve into each step, providing a clear understanding of how the YOLOv8 model was constructed and evaluated for basil crop detection.

### Step-by-Step Guide

**1. Project Setup**

* **Environment:**
    1. Ensure you have Python (version 3.7 or higher) installed on your system.
    2. Set up a virtual environment to manage project dependencies effectively. Tools like `venv` or `conda` can be used. Refer to their documentation for specific instructions.
* **Libraries:**
    1. Install the necessary Python libraries using `pip`:
        * `seaborn` (for statistical data visualization)
        * `scikit-learn` (for machine learning algorithms and tools)
        * `opencv-python` (for image processing)
        * `ultralytics` (the object detection model)

**2. Data Acquisition/Collection**

* **Basil Crop Dataset:**
    1. The dataset containing images of basil crops in fields. (around 209 images)
    2. The dataset include annotations for each basil plant, typically in the form of bounding boxes around the plants. (in YOLO format)

**3. Data Pre-processing**

* **Resizing Images and Annotations:**
    1. Resizing all the images in the dataset to a uniform size, such as 640 x 480 pixels (YOLOv8 image format).
    2. Adjust the corresponding annotations (bounding boxes) to reflect the resized image dimensions.
* **Data Normalization:**
    1. Normalize the pixel values of resized images to a common scale (e.g., [0, 1] range) to ensure consistency in input data for the model.
    2. Typical normalization techniques include dividing pixel values by 255 (for 8-bit images) or using z-score normalization based on dataset statistics.
* **Data Split:**
    1. Divide your dataset into two distinct subsets: a training set (typically around 80% of the data) and a validation set (remaining 20%). (171 images for training and 38 images for testing)
    2. The training set is used to train the YOLOv8 model, while the validation set is used to evaluate its performance on unseen data.

**4. Model Building and Evaluation**

* **YOLOv8 Model Configuration:**
    1. Utilize the `yolov8` library to configure and build the YOLOv8 model.
    2. Specify the model architecture (e.g., YOLOv8s, YOLOv8m, etc.). (we are using YOLOv8n)
    3. Defining the training parameters such as batch size, learning rate, optimizer, etc.).
    4. Adapt the model to detect basil crops by modifying the class labels. Refer to the `yolov8` documentation for detailed instructions.
* **Training:**
    1. Train the YOLOv8 model using the prepared training dataset.
    2. The model learns to identify basil crops within the images based on the provided annotations.
    3. Training typically involves iterating through the training data multiple times.

* **Performance Metrics:**
    1. Evaluate the model's performance on the validation set using metrics like:
        * Mean Average Precision (mAP) at different IoU thresholds.
        * Precision and Recall.
        * F1 Score.
    2. Analyze training and validation loss values (box loss and classification loss).

[![DOI](https://zenodo.org/badge/799295896.svg)](https://zenodo.org/doi/10.5281/zenodo.11262948)
