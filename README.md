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
- **Encoder**: CNN (ResNet) to extract image features
- **Decoder**: LSTM + Attention (Bahdanau Attention) to generate captions based on image features

## 📦 Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/NguyenPhuongDng/image-caption.git
    cd image-caption
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    ```

3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the trained model files** from Kaggle:
    - **Model download link**: [Kaggle Model](https://www.kaggle.com/models/donghip/image-caption)

5. After downloading, place the model files (`model.keras` and `vocab.pkl`) into the `model/` folder:
    ```
    image_caption/
    └── model/
        ├── model.keras
        └── vocab.pkl
    ```

## ▶️ Run the App
1. Navigate to the `app/` directory:
    ```bash
    cd app
    ```

2. Run the Flask app:
    ```bash
    python main.py
    ```

3. Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the app.

## 📄 License
MIT License
