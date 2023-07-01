# EmoLearn
Emotion Detection from Images and Webcam

![Python](https://img.shields.io/badge/-Python-black?style=flat&logo=python)
![Deep Learning](https://img.shields.io/badge/-Deep%20Learning-566be8?style=flat)
![Tensorflow](https://img.shields.io/badge/-Tensorflow-gray?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/-Keras-gray?style=flat&logo=keras)
![OpenCV](https://img.shields.io/badge/-OpenCV-gray?style=flat&logo=opencv)
![NumPy](https://img.shields.io/badge/-NumPy-gray?style=flat&logo=numpy)
![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-black?style=flat&logo=jupyter)
![Streamlit](https://img.shields.io/badge/-Streamlit-f0806c?style=flat)


## Description
An emotion detection system detect and classify emotions using deep learning techniques (CNN).
It trained a model that can accurately recognize and classify human emotions, acting like a human brain to detect emotions.
And Innovated to analyze and recognize human emotions in real-time using a webcam.
EmoLearn aims to interpret and categorize emotions into seven distinct categories, including happy, sad, anger, fear, disgust, surprise, and neutral.

## Steps taken in this project <a name="project-steps"></a>
- Planning
- Data Collection and Preparation
- Evaluation of model
- User Interface Development for real time usage
- Deployment on streamlit-sharing

  
## Installation requirements 
To run this code, you need to:
- import streamlit as st
- import numpy as np
- import cv2
- import tensorflow as tf
- import os
- from keras.models import model_from_json
- from PIL import Image
- from google.colab import drive


## Labelling 
| Label | Description |
| --- | --- |
| 0 | angry |
| 1 | disgust |
| 2 | fear |
| 3 | happy |
| 4 | neutral |
| 5 | sad |
| 6 | surprise |


## Further development
* [ ]  **Model optimization**:To extend the model to be more culturally sensitive and adaptable
* [ ]  **Dataset expansion**:To expand and diversify the dataset to ensure it represents a wide range of cultures, and expressions.
* [ ]  **Data augmentation**:To improve performance in challenging environments, such as low-light conditions or low-resolution images
* [ ]  **Color image support**:To modify model to handle color images as input.


## Refrences
- Emotion detection: https://github.com/juan-csv/emotion_detection
- Face Recognition: https://github.com/juan-csv/face-recognition
- Haar Cascade: https://github.com/opencv/opencv/tree/master/data/haarcascades
- Dataset from Google, Github, ImageNet
- Open Source Libraires
