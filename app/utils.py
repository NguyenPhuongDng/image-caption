import tensorflow as tf
import pickle
import numpy as np
import re
from PIL import Image
from model import *

def load_model_and_vocab(model_path, vocab_path):
    model = tf.keras.models.load_model(model_path)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return model, vocab

def genarate_caption(image_stream, model, temperature=1.0, token= 50):
    image = Image.open(image_stream).convert("RGB")
    #image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.cast(image, tf.float32) / 255.0
    image = np.array(tf.expand_dims(image, axis=0))
    #print(image.shape)
    caption = model.predict(image, temperature=temperature, token=token)
    return caption

def idx_to_caption(idx, vocab):
    # Chuyển idx thành list nếu là tensor/mảng
    if isinstance(idx, tf.Tensor):
        idx = idx.numpy().tolist()  # Chuyển tensor thành list
    elif isinstance(idx, np.ndarray):
        idx = idx.tolist()
        
    tokens = vocab.ids_to_text(idx)  # Chuyển danh sách ID thành danh sách từ
    str_caption = " ".join(tokens)
    return str_caption

def process_caption(caption, unk_token="<UNK>", start_token="startseq", end_token="endseq"):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", " ", caption).split()
    
    filtered_tokens = []
    for token in caption:
        if token == end_token:
            break
        if token not in {unk_token, start_token, end_token}:
            filtered_tokens.append(token)
    
    return " ".join(filtered_tokens).strip()

def predict_caption(image_path, model, vocab, temperature=1.0, token=50):
    caption = genarate_caption(image_path, model, temperature=temperature, token=token)
    caption = idx_to_caption(caption, vocab)
    caption = process_caption(caption)
    return caption
