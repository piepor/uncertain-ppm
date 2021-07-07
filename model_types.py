import numpy as np
import tensorflow as tf
from model_utils import DecoderLayer, positional_encodings


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
            maximum_position_encoding, embedding_path, trainable, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if embedding_path is not None:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size, 
                    d_model, weights=[np.load(embedding_path)], trainable=trainable)
        else:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size, 
                    d_model, trainable=trainable)
        self.pos_encoding = positional_encodings(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, training, look_ahead_mask)

            attention_weights['decoder_layer{}_block'.format(i+1)] = block1
    
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
            pe_target, embedding_path, trainable, rate=0.1):
        super(Transformer, self).__init__()

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                target_vocab_size, pe_target, embedding_path, trainable, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, tar, training, look_ahead_mask):

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, training, look_ahead_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
