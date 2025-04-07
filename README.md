# 🖼️ Image Captioning Web App

A web application that automatically generates captions for uploaded images using deep learning.

## 🚀 Overview
This project allows users to upload an image and receive a natural language description, for example:
> Image: a dog running on grass → Caption: "A dog is running on the grass."

The app is built with:
- `Flask` for the web server
- `TensorFlow/Keras` for the deep learning model
- `ResNet` or `InceptionV3` + `LSTM` as the encoder-decoder architecture

## 🧠 Model Architecture
- **Encoder**: CNN to extract image features
- **Decoder**: LSTM to generate captions based on image features

