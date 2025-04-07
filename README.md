# 🖼️ Image Captioning Web App

A web application that automatically generates captions for uploaded images using deep learning.

## 🚀 Overview
This project allows users to upload an image and receive a natural language description, for example:
> Image: a dog running on grass → Caption: "A dog is running on the grass."

The app is built with:
- `Flask` for the web server
- `TensorFlow/Keras` for the deep learning model
- `ResNet` + `Bahdanau Attention` + `LSTM` as the encoder-decoder architecture

## 🧠 Model Architecture
- **Encoder**: CNN to extract image features
- **Decoder**: LSTM + Attention to generate captions based on image features

## 📦 Setup Instructions
```bash
git clone https://github.com/NguyenPhuongDng/image-caption.git
cd image-caption
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

You need to download the trained model files
link dowload model : https://www.kaggle.com/models/donghip/image-caption
After downloading, place them in the model/ folder:
image_caption/
└── model/
    ├── model.keras
    └── vocab.pkl

## ▶️ Run the App
