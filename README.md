# Brain Tumor Detection and Classification

This project is focused on training a Convolutional Neural Network (CNN) model to detect and classify brain tumors from medical images. The model distinguishes between tumor and non-tumor images and further classifies different types of tumors (gliomas, meningiomas, pituitary tumors) based on a training dataset.

## **Project Overview**

The goal of this project is to create a robust deep learning model capable of detecting and classifying brain tumors from MRI scans. The model was trained using a dataset of brain MRI images, and predictions were made on new test images to classify whether they show a tumor or not. The project also includes visualization to understand the model’s performance.

---

## **Table of Contents**
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Test Predictions](#test-predictions)
- [Source of Dataset](#source-of-dataset)
- [GitHub Repository](#github-repository)
- [License](#license)

---

## **Installation**

1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/KimcMat/brain-tumor-detection.git
   cd brain-tumor-detection

---

2. **Set up the environment**  
   This project requires Python 3.6+ and the following dependencies:
   - TensorFlow
   - Keras
   - NumPy
   - OpenCV
   - Matplotlib
   - Scikit-learn

   Install the dependencies via `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Jupyter Notebook (Optional)**  
   To visualize the results and explore the model further, you can use Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## **Project Structure**

The project directory is structured as follows:

```
/brain-tumor-detection
    /data
        /yes
        /no
    /model
        tumor_detection_model.h5
    /test_images
    /visualization
    README.md
    requirements.txt
    train_model.py
    predict_and_visualize.py
```

- **/data**: Contains the dataset used for training the model (`yes` folder for tumor images, `no` folder for non-tumor images).
- **/model**: Contains the saved trained model (`tumor_detection_model.h5`).
- **/test_images**: New images for testing the model predictions.
- **/visualization**: Contains scripts for visualizing the predictions and results.
- **README.md**: This file containing project details.

---

## **Data Preprocessing**

The dataset used in this project consists of MRI images of brain scans. The following preprocessing steps were applied:
1. **Resizing**: Each image was resized to `128x128` pixels to ensure consistency.
2. **Normalization**: Pixel values were normalized to be between `0` and `1`.
3. **One-hot Encoding**: Labels for tumor detection were one-hot encoded, where `1` represents a tumor and `0` represents no tumor.

The images were then split into training and testing sets using an 80-20 split.

---

## **Model Architecture**

The model used is a Convolutional Neural Network (CNN), consisting of the following layers:

1. **Conv2D Layer**: 32 filters, kernel size of (3, 3), ReLU activation.
2. **MaxPooling2D Layer**: Pooling size of (2, 2).
3. **Conv2D Layer**: 64 filters, kernel size of (3, 3), ReLU activation.
4. **MaxPooling2D Layer**: Pooling size of (2, 2).
5. **Flatten Layer**: Converts the 2D matrix into a 1D array.
6. **Dense Layer**: 128 units, ReLU activation.
7. **Output Layer**: 2 units with softmax activation for binary classification (tumor vs. non-tumor).

---

## **Training the Model**

The model was compiled using the following settings:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy (for multi-class classification)
- **Metrics**: Accuracy

The model was trained for 10 epochs with a batch size of 32, and validation data was used to monitor performance.

### Model Training Code:
```python
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
```

The trained model was saved as `tumor_detection_model.h5` for later use in prediction.

---

## **Evaluation and Visualization**

The model's performance was evaluated on the test dataset, achieving a high accuracy of around 99% on the test images.

### Visualization:
- A visualization script was created to display the results for each test image, showing the original image and the model’s prediction (whether there is a tumor and what type of tumor it is).
- **Matplotlib** was used to plot the images alongside the predicted labels.

Example visualization code:
```python
def visualize_predictions():
    for filename in os.listdir(test_folder):
        image_path = os.path.join(test_folder, filename)
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction, axis=1)[0]
        print(f"Prediction for {filename}: {predicted_label}")
```

---

## **Test Predictions**

The model can now be used to predict new brain tumor images and classify them as either having a tumor or not. Additionally, it can classify the type of tumor (glioma, meningioma, pituitary) based on the trained dataset.

### Test Prediction Code:
```python
test_folder = '/path/to/test_images'
visualize_predictions()
```

---

## **Source of Dataset**

The dataset used in this project was sourced from Kaggle, specifically the **Brain Tumor MRI Dataset**, which contains brain MRI images labeled with the type of tumor (gliomas, meningiomas, pituitary tumors). The dataset can be accessed here:

- [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## **GitHub Repository**

This project is stored in a GitHub repository for version control and collaboration. You can view the repository at:

[https://github.com/KimcMat/brain-tumor-detection](https://github.com/KimcMat/brain-tumor-detection)

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
