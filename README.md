# Skin Cancer Detection Using YOLOv8

## Overview

This project presents a YOLOv8-based skin cancer detection system designed to identify and classify seven different skin lesion categories.

The model was trained using transfer learning on a custom skin lesion dataset and evaluated using standard object detection metrics.

## Classes

* Melanoma
* Nevus
* BCC
* AK
* BKL
* DF
* VASC

## Model

* YOLOv8n
* Fine-Tuned on Skin Lesion Dataset

## Training Configuration

* Epochs: 165
* Image Size: 800 × 800
* Batch Size: 16
* Learning Rate: 0.01
* Dropout: 0.15

## Results

| Metric    | Value  |
| --------- | ------ |
| Precision | 0.7354 |
| Recall    | 0.5646 |
| mAP@50    | 0.6472 |
| mAP@50-95 | 0.5399 |

## Project Files

* train.py
* preprocessing.py
* predict_pipeline.py
* model_arch.py
* local_gui.py

## Technologies Used

* Python
* YOLOv8
* OpenCV
* NumPy
* Pandas
* Matplotlib
* CustomTkinter

## Course

Artificial Intelligence / Computer Vision Semester Project
