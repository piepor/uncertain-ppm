import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


class PreprocessingModel(tf.keras.Model):
    def __init__(self, features, ds):
        super().__init__()

        self.preprocessing_layers = []
        for feature in features:
            if feature['feature-type'] == 'string':
                string_lookup = tf.keras.layers.StringLookup(
                    vocabulary=feature['vocabulary'], num_oov_indices=1)
                self.preprocessing_layers.append(
                    tf.keras.Sequential([
                        string_lookup,
                        tf.keras.layers.Embedding(
                            input_dim=feature['input-dim'],
                            output_dim=feature['output-dim']
                        )
                    ], name=feature['name']
                    )
                )
            elif feature['feature-type'] == 'continuous':
                normalization_layer = tf.keras.layers.Normalization(axis=None)
                normalization_layer.adapt(ds.map(lambda x: x[feature['name']]))
                self.preprocessing_layers.append(
                    tf.keras.Sequential([
                        normalization_layer,
                        tf.keras.layers.Reshape((-1, 1))
                    ], name=feature['name'])
                )
            elif feature['feature-type'] == 'categorical':
                self.preprocessing_layers.append(
                    tf.keras.Sequential([
                        tf.keras.layers.Embedding(
                            input_dim=feature['input-dim'],
                            output_dim=feature['output-dim']
                        )
                    ], name=feature['name']
                    )
                )
    def call(self, inputs):
        return tf.concat(
            [self.preprocessing_layers[i](input_feat) for i, input_feat in enumerate(inputs)], 
            axis=-1)

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangule, countin from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, 
                key_dim=embed_dim)
        self.ffn = keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(
                batch_size, seq_len, seq_len, tf.bool)
        attn_output = self.att(inputs, inputs, 
                attention_mask=causal_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class GeneralModel(tf.keras.Model):
    def __init__(self, num_layers, features, ds, num_heads, 
            feed_forward_dim, maximum_positional_embedding=10000):
        super().__init__()
        num_voc = features[0]['input-dim']
        embed_dim = sum([feature['output-dim'] for feature in features])
        self.embedding_model = PreprocessingModel(features, ds)
        self.pos_emb = positional_encoding(maximum_positional_embedding, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
        self.transformer = tf.keras.Sequential()
        self.ffn_output = tf.keras.layers.Dense(num_voc)
        for n in range(num_layers):
            self.transformer.add(self.transformer_block)

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        # add positional embedding
        seq_len = tf.shape(feature_embedding)[1]
        feature_embedding += self.pos_emb[:, :seq_len, :]
        out = self.transformer(feature_embedding)
        out = self.ffn_output(out)
        return out

class ModelWithTemperature(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        # freeze weights
        self.inner_model.trainable = False
        assert self.inner_model.trainable == False
        self.temperature_scale = tf.Variable(1.5)

    def call(self, inputs):
        logits = self.inner_model(inputs)
        return tf.math.divide(logits, self.temperature_scale)

