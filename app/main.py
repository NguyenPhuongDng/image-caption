from flask import Flask, request, render_template
from io import BytesIO
import base64
from utils import load_model_and_vocab, predict_caption
from model import *

app = Flask(__name__)
model_path = 'app/model/image_caption_attention_model.keras'
vocab_path = 'app/model/vocab.pkl'
model, vocab = load_model_and_vocab(model_path, vocab_path)

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_data = None
    if request.method == "POST":
        image = request.files["image"]
        if image:
            image_bytes = image.read()
            image_stream = BytesIO(image_bytes)
            caption = predict_caption(image_stream, model, vocab, temperature=0.5, token=50)

            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
            mime_type = image.mimetype
            image_data = f"data:{mime_type};base64,{encoded_image}"

    return render_template("index.html", caption=caption, image_data=image_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
