# Project Title
Pneumonia Detection Using Deep Learning and Chest X-rays
## Overview
This repository contains the implementation of a state-of-the-art approach to pneumonia detection from chest X-ray images using Convolutional Neural Networks (CNNs).
Our project leverages advanced deep learning techniques to develop an accurate, robust, and efficient model capable of automating the diagnosis of pneumonia.

## Project Description
Pneumonia is a major global health issue that requires precise and timely diagnosis to facilitate effective treatment.
Leveraging the growing availability of large datasets and the computational power of modern GPUs, this project employs sophisticated CNN architectures to address this medical challenge.

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


