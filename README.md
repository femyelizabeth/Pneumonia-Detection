# Pneumonia Detection Using Deep Learning and Chest X-rays
This repository contains the source code and documentation for a deep learning project aimed at detecting pneumonia from chest X-ray images. Developed using advanced convolutional neural networks (CNNs), this project leverages state-of-the-art deep learning methodologies and datasets to provide a high-accuracy tool for healthcare professionals.

## Overview
Pneumonia is a significant health problem worldwide, leading to high morbidity and mortality rates. Early and accurate detection is crucial for effective treatment and patient recovery. Traditional diagnostic methods rely heavily on the expertise of radiologists to interpret chest X-rays, which can sometimes lead to variability in diagnosis accuracy.

In response to this challenge, our project utilizes machine learning algorithms to automate the detection process, reducing the dependency on manual interpretation and aiming to enhance diagnostic accuracy. We employ several pre-trained models such as VGG-16, Inception V3, and ResNet, fine-tuned on a large dataset of chest X-ray images to distinguish between normal and pneumonia-affected lungs.

## Project Description
Pneumonia is a major global health issue that requires precise and timely diagnosis to facilitate effective treatment.
Leveraging the growing availability of large datasets and the computational power of modern GPUs, this project employs sophisticated CNN architectures to address this medical challenge.

## Key Features

- **Automated Detection**: Simplify the process of identifying pneumonia in chest X-rays, making it faster and potentially more accurate.
- **Deep Learning Powered**: Utilizes cutting-edge CNN architectures that have been fine-tuned for high accuracy and performance.
- **Accessible and Open**: All source code, models, and training procedures are open-source and available for community use and improvement.
- **Educational Value**: Provides a resource for learning and experimentation for those interested in deep learning and medical image analysis.

This project is intended for educational purposes, to assist medical professionals, and to foster further research in automated medical image diagnosis. Whether you are a developer, a data scientist, a medical professional, or a student, this project provides valuable insights into the application of deep learning techniques in healthcare.

### Methodology
The methodology involves:

Data Acquisition: Utilizing a Kaggle dataset comprising 5863 chest X-ray images divided into 'PNEUMONIA' and 'NORMAL' categories.
Data Augmentation: To overcome the limitations of dataset size and improve model generalizability.
Transfer Learning: Using pre-trained models such as VGG-16, Inception V3, and ResNet.
Fine-Tuning: Adjusting hyperparameters and layers of pre-trained models to better suit pneumonia detection.
Weighted Classification: Combining predictions from various models to enhance accuracy and reliability.

### Data
The dataset used can be found on Kaggle, consisting of training, testing, and validation subsets categorized into PNEUMONIA and NORMAL classes. 
This dataset is crucial for training and evaluating the performance of our models.
[Download Training Dataset from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Installation

To get started with this project, you'll need to clone the repository and install the necessary dependencies. Follow these steps:

```bash
# Clone the repository
git clone https://github.com/femyelizabeth/Pneumonia-Detection.git

# Change directory to the project folder
cd Pneumonia-Detection

# Install required Python packages
pip install -r requirements.txt

# Run the Flask app
python app.py
```

## Pre-trained Models

The models used in this project have been trained and saved on Kaggle due to their large size. You can download the models from the following Kaggle dataset:

[Download Trained Models from Kaggle](https://www.kaggle.com/datasets/femyelizabeth/model-files/data)

### How to Use the Models

After downloading the models, you can load them in your Python environment using the appropriate library commands. For example, if you are using TensorFlow/Keras, you can load a model like this:

```python
from tensorflow import keras
model = keras.models.load_model('path/to/downloaded/model.h5')


