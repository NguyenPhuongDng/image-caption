import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class Vocabulary:
    def __init__(self, captions, oov_token="<UNK>"):
        self.tokenizer = Tokenizer(oov_token=oov_token)
        self.tokenizer.fit_on_texts(captions)
        self.word_index = self.tokenizer.word_index
        self.index_word = {v: k for k, v in self.word_index.items()}
        self.vocab_size = len(self.word_index) + 1
        self.oov_token = oov_token
    
    def text_to_ids(self, text: tf.Tensor, max_len: int):
        text = text.numpy().decode("utf-8")  # Chuyển từ Tensor thành string
        seq = self.tokenizer.texts_to_sequences([text])[0]
        padded_seq = pad_sequences([seq], maxlen=max_len, padding="post")[0]
        return tf.convert_to_tensor(padded_seq, dtype=tf.int32)

    def ids_to_text(self, ids):
        if isinstance(ids, tf.Tensor):  # Kiểm tra nếu ids là Tensor
            ids = ids.numpy().tolist()  # Chuyển sang danh sách

        if isinstance(ids, np.ndarray):  # Nếu là NumPy array
            ids = ids.tolist()
            
        words = [self.index_word.get(int(i), self.oov_token) for i in ids]
        return words
        #"".join(words).replace(self.oov_token, "")

    def vocab_size(self):
        return self.vocab_size


@tf.keras.utils.register_keras_serializable(package="MyLayers", name="Encoder")
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.ResNet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
        self.ResNet.trainable = False
        self.pooling = layers.GlobalAveragePooling2D()
        self.norm = layers.LayerNormalization(axis=-1)
        
    def call(self, x):
        x = self.ResNet(x)
        x = tf.reshape(x, [tf.shape(x)[0], x.shape[1]*x.shape[2], x.shape[-1]])
        x = self.norm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls()

    def build(self, input_shape):
        self.ResNet.build(input_shape)
        resnet_output_shape = self.ResNet.compute_output_shape(input_shape)
        reshaped_shape = [resnet_output_shape[0], resnet_output_shape[1] * resnet_output_shape[2], resnet_output_shape[3]]
        self.norm.build(reshaped_shape)
        self.built = True
        
    def compute_output_shape(self, input_shape):
        x = self.ResNet.compute_output_shape(input_shape)
        x = self.pooling.compute_output_shape(x)
        return x
        
@tf.keras.utils.register_keras_serializable(package="MyLayers", name="Attention")
class Attention(tf.keras.Model):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W_a = layers.Dense(hidden_size, use_bias=False)  # Biến đổi h_t
        self.U_a = layers.Dense(hidden_size, use_bias=False)  # Biến đổi h_s
        self.V_a = layers.Dense(1, use_bias=False)  # Tính attention scores
        self.W_c = layers.Dense(self.hidden_size, activation="tanh")  # Biến đổi đầu ra
        self.norm = layers.LayerNormalization(axis=-1)

    def call(self, h_t, h_s):
        # h_t: (None, 512), h_s: (None, 49, 2048)
        
        # Biến đổi h_t thành không gian hidden_size
        h_t_transformed = self.W_a(h_t)  # Shape: (None, hidden_size)
        h_t_transformed = tf.expand_dims(h_t_transformed, axis=1)  # Shape: (None, 1, hidden_size)
        
        # Biến đổi h_s thành không gian hidden_size
        h_s_transformed = self.U_a(h_s)  # Shape: (None, 49, hidden_size)
        
        # Tính attention scores (additive attention)
        attention_input = tf.tanh(h_s_transformed + h_t_transformed)  # Shape: (None, 49, hidden_size)
        attention_scores = self.V_a(attention_input)  # Shape: (None, 49, 1)
        
        # Tính attention weights
        alpha = tf.nn.softmax(attention_scores, axis=1)  # Shape: (None, 49, 1)
        
        # Tính context vector
        context_vector = tf.reduce_sum(alpha * h_s, axis=1)  # Shape: (None, 2048)
        
        # Kết hợp context_vector với h_t
        concat_vector = tf.concat([context_vector, h_t], axis=-1)  # Shape: (None, 2048 + 512) = (None, 2560)
        
        # Tạo attention vector và chuẩn hóa
        attention_vector = self.W_c(concat_vector)  # Shape: (None, hidden_size)
        attention_vector = self.norm(attention_vector)  # Shape: (None, hidden_size)
        
        return attention_vector  

    def build(self, input_shape):
        h_t_shape, h_s_shape = input_shape  # h_t: (None, 512), h_s: (None, 49, 2048)
        
        self.W_a.build(h_t_shape)  # Input: (None, 512), Output: (None, hidden_size)
        self.U_a.build(h_s_shape)  # Input: (None, 49, 2048), Output: (None, 49, hidden_size)
        self.V_a.build([h_t_shape[0], 49, self.hidden_size])  # Input: (None, 49, hidden_size), Output: (None, 49, 1)
        
        concat_shape = [h_t_shape[0], h_s_shape[-1] + h_t_shape[-1]]  # (None, 2048 + 512) = (None, 2560)
        self.W_c.build(concat_shape)  # Input: (None, 2560), Output: (None, hidden_size)
        
        self.norm.build([concat_shape[0], self.hidden_size])  # Input: (None, hidden_size)        
        self.built = True

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "hidden_size": self.hidden_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(hidden_size=config["hidden_size"])


@tf.keras.utils.register_keras_serializable(package="MyLayers", name="Decoder")
class Decoder(tf.keras.Model):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.embed = layers.Embedding(self.vocab_size, self.embed_size)
        
        if num_layers == 1:
            self.lstm = layers.LSTM(self.hidden_size,
                                    return_sequences=False,
                                    return_state=True)
        else:
            lstm_layers = []
            for i in range(self.num_layers):
                return_seq = True if i < self.num_layers - 1 else False
                lstm_layers.append(
                    layers.LSTM(self.hidden_size,
                                return_sequences=return_seq,
                                return_state=True)
                )
            self.lstm = tf.keras.Sequential(lstm_layers)
            
        self.norm1 = layers.LayerNormalization(axis=-1)
        self.norm2 = layers.LayerNormalization(axis=-1)
        self.linear = layers.Dense(self.vocab_size)
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(0.3)
        self.softmax = layers.Softmax()

    def call(self, encoder_features, captions, hidden_state):
        embeddings = self.embed(captions)

        combined = tf.concat([encoder_features, embeddings], axis=-1)
        combined = self.norm1(combined)
        combined = tf.expand_dims(combined, axis=1)
        
        if self.num_layers == 1:
            lstm_out, h, c = self.lstm(combined,
                                        initial_state=hidden_state)
        else:
            lstm_out = combined
            states = []
            for i, layer in enumerate(self.lstm.layers):
                if i == 0:
                    lstm_out, h, c = layer(lstm_out,
                                          initial_state=hidden_state)
                else:
                    lstm_out, h, c = layer(lstm_out, 
                                          initial_state=[h, c])
                    
        output = self.relu(lstm_out)
        output = self.dropout(output)
        output = self.norm2(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output, [h, c]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(
            embed_size=config["embed_size"],
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"]
        )
        
    def build(self, input_shape):
        captions_shape, hidden_state_shape = input_shape
        bsz = captions_shape[0]
        self.embed.build(captions_shape)
        
        combined_dim = self.hidden_size + self.embed_size
        combined_shape = tf.TensorShape([bsz, 1, combined_dim])
        
        self.norm1.build(tf.TensorShape([bsz, combined_dim]))
        
        if self.num_layers == 1:
            self.lstm.build(combined_shape)
        else:
            current_shape = combined_shape
            for i, lstm_layer in enumerate(self.lstm.layers):
                lstm_layer.build(current_shape)
                if i < self.num_layers - 1:
                    current_shape = tf.TensorShape([current_shape[0], current_shape[1], self.hidden_size])
                else:
                    current_shape = tf.TensorShape([current_shape[0], self.hidden_size])
        
        self.norm2.build(tf.TensorShape([bsz, self.hidden_size]))
        self.linear.build(tf.TensorShape([bsz, self.hidden_size]))
        self.relu.build(tf.TensorShape([bsz, self.hidden_size]))
        self.dropout.build(tf.TensorShape([bsz, self.hidden_size]))
        self.softmax.build(tf.TensorShape([bsz, self.vocab_size]))
        
        self.built = True
        
@tf.keras.utils.register_keras_serializable(package="MyLayers", name="Image_caption")        
class Image_caption(tf.keras.Model):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, **kwargs):
        super(Image_caption, self).__init__(**kwargs)
        self.embed_size = embed_size  
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size 
        self.num_layers = num_layers 

        self.encoder = Encoder()
        self.decoder = Decoder(self.embed_size, self.vocab_size, self.hidden_size, self.num_layers)
        self.attention = Attention(hidden_size)

        self.hidden_state_start = [
            tf.Variable(tf.random.normal([1, self.hidden_size]), trainable=True, name="h0"),
            tf.Variable(tf.random.normal([1, self.hidden_size]), trainable=True, name="c0")
        ]
        
    def call(self, images_captions):
        output_train = []
        
        images, captions = images_captions
        
        decoder_input = captions[:, 0]
        decoder_input = tf.cast(decoder_input, dtype=tf.int32)
   
        encoder_features = self.encoder(images)
        bsz = tf.shape(encoder_features)[0]
        
        hidden_state = [
            tf.tile(self.hidden_state_start[0], [bsz, 1]),
            tf.tile(self.hidden_state_start[1], [bsz, 1])
        ]

        for di in range(1, captions.shape[-1]):
            attention_vector = self.attention(hidden_state[0], encoder_features)
            output, hidden_state = self.decoder(attention_vector, decoder_input, hidden_state)
            output_train.append(output)
            
            decoder_input = captions[:,di]
        
        return tf.stack(output_train, axis=1)
        
    def predict(self, images, temperature=1.0, token=50):
        seq_predicted = []
        
        #images = [1, 224, 224, 3]
        decoder_input = tf.constant([3], dtype=tf.int32)
        #decoder_input = tf.expand_dims(decoder_input, axis=0)

        encoder_features = self.encoder(images)
        bsz = tf.shape(encoder_features)[0]
        
        seq_predicted.append(decoder_input)
        hidden_state = [
            tf.tile(self.hidden_state_start[0], [bsz, 1]),
            tf.tile(self.hidden_state_start[1], [bsz, 1])
        ]

        for di in range(1, token):
            attention_vector = self.attention(hidden_state[0], encoder_features)
            output, hidden_state = self.decoder(attention_vector, decoder_input, hidden_state)
            if temperature == 0:
                word_predicted = tf.math.argmax(output, -1)
            else:
                logits = tf.math.log(output + 1e-10) / temperature
                word_predicted = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)[:, 0]
            
            seq_predicted.append(word_predicted)
            decoder_input = word_predicted
                
        return seq_predicted
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "hidden_state_h": self.hidden_state_start[0].numpy().tolist(),
            "hidden_state_c": self.hidden_state_start[1].numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Trích xuất các thông số cần thiết cho hàm __init__
        embed_size = config.pop("embed_size")
        hidden_size = config.pop("hidden_size")
        vocab_size = config.pop("vocab_size")
        num_layers = config.pop("num_layers")
        
        # Tạo model mới
        model = cls(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers
        )
        
        # Khôi phục giá trị hidden_state_start
        if "hidden_state_h" in config and "hidden_state_c" in config:
            h_value = tf.constant(config["hidden_state_h"], dtype=tf.float32)
            c_value = tf.constant(config["hidden_state_c"], dtype=tf.float32)
            model.hidden_state_start[0].assign(h_value)
            model.hidden_state_start[1].assign(c_value)
            
        return model

    def build(self, input_shape):
        images_shape, caption_shape = input_shape
        self.encoder.build(images_shape)
        encoder_output_shape = self.encoder.compute_output_shape(images_shape)
        attention_input_shape = ([images_shape[0], self.hidden_size], encoder_output_shape)
        self.attention.build(attention_input_shape)
        self.decoder.build([tf.TensorShape([None]), tf.TensorShape([None, self.hidden_size])])
        self.built = True

@tf.keras.utils.register_keras_serializable(package="MyFn", name="custom_loss")
def custom_loss(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    #loss = tf.cast(loss, tf.float32)
    mask = (y_true != 0) & (loss < 1e8) 
    mask = tf.cast(mask, loss.dtype)
    
    loss = loss*mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    
    return loss

@tf.keras.utils.register_keras_serializable(package="MyFn", name="masked_acc")
def masked_acc(labels, preds):
    mask = tf.cast(labels!=0, tf.float32)
    preds = tf.argmax(preds, axis=-1)
    labels = tf.cast(labels, tf.int64)
    match = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(match*mask)/tf.reduce_sum(mask)
    #acc = tf.cast(acc, tf.float32)
    return acc