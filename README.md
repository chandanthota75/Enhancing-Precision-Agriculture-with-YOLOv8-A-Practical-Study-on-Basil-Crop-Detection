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

**6. Inference and Visualization**

Once the model was successfully trained, we subjected it to rigorous evaluation using the validation dataset. The evaluation process involved running the model on the validation dataset and assessing its performance against the ground truth annotations. The results were meticulously calculated and presented, offering insights into the model's precision, recall, f1 score and various other metrics that gauge its proficiency in basil crop detection.

**Validation Results on Evaluation Metrics**

| Metric    | Precision | Recall | F1 Score | mAP50 | mAP50-95 |
|-----------|-----------|--------|----------|-------|----------|
| Result    | 0.960     | 0.987  | 0.969    | 0.983 | 0.895    |

The above concise table summarizes the major validation metrics, highlighting the model's precision, recall, F1 Score, mAP50, and mAP50-95 results. These metrics collectively attest to the YOLOv8 model's exceptional performance in basil crop detection, underpinning the core findings of your research.

**Validation Results on Bounding Boxes**

| Metric                | Train | Validation |
|-----------------------|-------|------------|
| Box Loss              | 0.428 | 0.435      |
| Classification Loss   | 0.377 | 0.435      |
| DFL Loss              | 0.891 | 0.778      |

The below table succinctly summarizes the training and validation losses, which are vital indicators of your YOLOv8 model's performance. These results provide insights into the efficiency of the model in terms of making predictions that closely align with the ground truth annotations.

In our study of basil crop detection with YOLOv8, we meticulously track the evolution of box losses, classification (cls) losses, and deformable (dfl) losses during the training and validation phases. The significance of these loss metrics lies in their ability to quantify various aspects of our model's performance. Train and validation box losses reflect the model's capacity to accurately localize basil crops within images, guiding the refinement of bounding box coordinates. Simultaneously, train and validation classification losses assess the model's aptitude for correctly categorizing basil crops, ensuring their distinction from background elements. Furthermore, the inclusion of deformable losses adds a layer of spatial adaptability, essential for handling the diverse growth stages and orientations of basil crops in the real-world context. By continuously monitoring these losses, we gain a nuanced perspective of our YOLOv8 model's precision and resilience, underscoring its suitability for the demanding realm of precision agriculture.

These loss metrics serve as fundamental pillars in our quest to optimize the YOLOv8 model for basil crop detection. They encapsulate the model's ability to precisely localize and classify basil crops, and the unique deformable losses enhance its spatial awareness. As we delve into the validation results, these losses provide granular insights into the model's accuracy and robustness, reinforcing its practical viability in real-world precision agriculture scenarios.

<div style="text-align: center;">
  <img src="images in paper/Basil Crop Losses.png" alt="Loss Evalution over Epochs" style="width: 700px;">
</div>

To comprehensively evaluate the performance of our YOLOv8 model in basil crop detection, we present a visual summary of true and predicted images with corresponding bounding box annotations. These annotated images highlight the precision and recall of our model by showcasing the accuracy with which it identifies basil crops within the dataset. The visual representation of the detection results serves as a vital component in assessing the real-world applicability of our approach, providing valuable insights into the model's ability to delineate basil crops across varying growth stages and orientations.

Including true and predicted images with annotations offers a tangible means of gauging the model's accuracy and efficiency in basil crop detection, strengthening the empirical evidence of our findings. The visual summary contributes to the overall transparency and interpretability of our research, enabling readers to assess the practical implications of our YOLOv8-based approach in precision agriculture.

This addition underscores the meticulous evaluation process and ensures a thorough comprehension of the model's performance, making our research more accessible and valuable to both the scientific community and practitioners in the field of precision agriculture.

<div style="text-align: center;">
  <img src="images in paper/Basil Crop Major Metrics.png" alt="Evaluation metrics over Epochs" style="width: 700px;">
</div>

## Conclusion

The confluence of deep learning technology, advanced agricultural practices, and cutting-edge machine vision systems has the potential to transform the landscape of precision agriculture. In this research, we embarked on a journey to revolutionize crop monitoring by focusing on basil, a high-value herb with distinct growth characteristics. Our endeavor was underpinned by a meticulous methodology that relied upon a well-curated dataset, rigorous model training, and meticulous validation. The results we obtained are testament to the promise and potential of this fusion of technology and agriculture.

The YOLOv8 model, meticulously configured and fine-tuned, emerged as a robust tool for basil crop detection. With training and validation losses of 0.428 and 0.435, respectively, the model showcases its adaptability to the diverse appearances of basil plants. The precision, recall, and F1 score metrics, standing at 0.960, 0.987, and 0.969, respectively, affirm its accuracy and efficiency in differentiating basil crops from their surroundings. Furthermore, the model's ability to maintain a mean average precision (mAP) of 0.983 at an intersection over union (IoU) threshold of 0.50 (mAP50) and 0.895 when assessed from 0.50 to 0.95 (mAP50-95) is a testament to its versatility

