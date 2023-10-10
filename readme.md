# Human Activity Recognition using LRCN and ConvLSTM

![Human Activity Recognition](img.png)

## Overview

This repository contains the code and resources for Human Activity Recognition (HAR) using Long-term Recurrent Convolutional Networks (LRCN) and Convolutional LSTM (ConvLSTM) models. The models are trained on the UCF50 dataset, a benchmark dataset for human action recognition, and tested on YouTube videos.

**Key Features:**

- Used UCF50 dataset, which contains 50 different human action classes. The models were trained on 7 classes.
- Tests the models on YouTube videos to evaluate real-world performance.

## Models

### Long-term Recurrent Convolutional Network (LRCN)

The LRCN model combines the spatial feature extraction capabilities of Convolutional Neural Networks (CNNs) with the temporal modeling abilities of Recurrent Neural Networks (RNNs). It has been trained to recognize human activities in video sequences.

### Convolutional LSTM (ConvLSTM)

The ConvLSTM model extends the traditional LSTM architecture by incorporating convolutional layers. It is designed to capture spatiotemporal dependencies in video data, making it suitable for activity recognition tasks.

## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Divyan-shu-Singh/HumanActivityRecognition_cnn_lstm.git

   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the provided Jupyter notebooks for testing the models on YouTube videos and evaluating their performance.

## Pre-trained Models

We provide pre-trained LRCN and ConvLSTM models in the models/ directory. You can use these models for quick evaluation on your own video data.

## Dataset

The models were trained on the [UCF50 dataset](https://www.crcv.ucf.edu/data/UCF50.rar), which contains a diverse set of human activities. You can download the dataset and use it for further experimentation.

