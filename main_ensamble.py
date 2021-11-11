import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
import yaml
import datetime
import glob
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('features', 
                    help='yaml file specifying training features (e.g. act_res_time.params)')

args = parser.parse_args()
features_name = args.features

ds_train = tfds.load('helpdesk', split='train[:70%]', shuffle_files=True)
ds_vali = tfds.load('helpdesk', split='train[70%:85%]')
ds_test = tfds.load('helpdesk', split='train[85%:]')

vocabulary = ['<START>',  '<END>','Resolve SW anomaly', 'Resolve ticket', 'RESOLVED', 
              'DUPLICATE', 'Take in charge ticket', 'Create SW anomaly',
              'Schedule intervention', 'VERIFIED', 'Closed', 'Wait',
              'Require upgrade', 'Assign seriousness', 'Insert ticket', 'INVALID']

def compute_features(file_path, vocabularies):
    with open(file_path, 'r') as file:
        features = list(yaml.load_all(file, Loader=yaml.FullLoader))
    for feature in features:
        if feature['feature-type'] == 'string':
            feature['vocabulary'] = vocabularies[feature['name']]
    return features


class PreprocessingModel(tf.keras.Model):
    def __init__(self, features, ds):
        super().__init__()

        self.preprocessing_layers = []
        #breakpoint()
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
        #breakpoint()
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
    def __init__(self, num_layers, features, ds, embed_dim, num_heads, feed_forward_dim, num_voc):
        super().__init__()
        self.embedding_model = PreprocessingModel(features, ds)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
        self.transformer = tf.keras.Sequential()
        self.ffn_output = tf.keras.layers.Dense(num_voc)
        for n in range(num_layers):
            self.transformer.add(self.transformer_block)

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        out = self.transformer(feature_embedding)
        out = self.ffn_output(out)
        return out

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def compute_input_signature(features):
    train_step_signature = []
    for feature in features:
        if feature['dtype'] == 'string':
            train_step_signature.append(tf.TensorSpec(shape=(None, None), dtype=tf.string))
        elif feature['dtype'] == 'int64': 
            train_step_signature.append(tf.TensorSpec(shape=(None, None), dtype=tf.int64))
        elif feature['dtype'] == 'int32': 
            train_step_signature.append(tf.TensorSpec(shape=(None, None), dtype=tf.int32))
    return train_step_signature

#features_name = 'act_res_day_week_time.params'
features = compute_features(features_name, {'activity': vocabulary})
train_step_signature = compute_input_signature(features)
output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary,
                                                 num_oov_indices=1)
num_heads = 2
embed_dim = sum([feature['output-dim'] for feature in features])
feed_forward_dim = 512
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam()
    
ensamble_list = glob.glob('models_ensamble/helpdesk/ensamble*')
num = 0
for ensamble in ensamble_list:
    #breakpoint()
    if int(ensamble.split('_')[2]) > num:
        num = int(ensamble.split('_')[2])
model_dir = 'models_ensamble/helpdesk/ensamble_{}'.format(int(num)+1)
os.makedirs(model_dir)
copyfile(features_name, os.path.join(model_dir, 'features.params'))

padded_shapes = {
    'activity': [None],
    'resource': [None],
    'product': [None],    
    'customer': [None],    
    'responsible_section': [None],    
    'service_level': [None],    
    'service_type': [None],    
    'seriousness': [None],    
    'workgroup': [None],
    'variant': [None],    
    'relative_time': [None],
    'day_part': [None],
    'week_day': [None]
}
padding_values = {
    'activity': '<PAD>',
    'resource': tf.cast(0, dtype=tf.int64),
    'product': tf.cast(0, dtype=tf.int64),    
    'customer': tf.cast(0, dtype=tf.int64),    
    'responsible_section': tf.cast(0, dtype=tf.int64),    
    'service_level': tf.cast(0, dtype=tf.int64),    
    'service_type': tf.cast(0, dtype=tf.int64),    
    'seriousness': tf.cast(0, dtype=tf.int64),    
    'workgroup': tf.cast(0, dtype=tf.int64),
    'variant': tf.cast(0, dtype=tf.int64),    
    'relative_time': tf.cast(0, dtype=tf.int32),
    'day_part': tf.cast(0, dtype=tf.int64),
    'week_day': tf.cast(0, dtype=tf.int64),
}

# train loop
epochs = 100
wait = 0
best = 0
patience = 10
batch_size = 32
num_models_ensamble = 5

padded_ds = ds_train.padded_batch(batch_size, 
        padded_shapes=padded_shapes,
        padding_values=padding_values).shuffle(buffer_size=10000, 
                                               reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
padded_ds_vali = ds_vali.padded_batch(batch_size, 
                                      padded_shapes=padded_shapes,
                                      padding_values=padding_values).prefetch(tf.data.AUTOTUNE)
#breakpoint()

for _ in range(num_models_ensamble):
    #tf.keras.backend.clear_session()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    vali_log_dir = 'logs/gradient_tape/' + current_time + '/validation'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    vali_summary_writer = tf.summary.create_file_writer(vali_log_dir)
    model_path = os.path.join(model_dir, current_time)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    vali_loss = tf.keras.metrics.Mean(name='vali_loss')
    vali_accuracy = tf.keras.metrics.Mean(name='vali_accuracy')

    model = GeneralModel(1, features, ds_train, embed_dim, num_heads, feed_forward_dim, features[0]['input-dim'])

    @tf.function(input_signature=[train_step_signature])
    def train_step(*args):
#def train_step(act, res, var):
#    input_data = [act[:, :-1], res[:, :-1], var[:, :-1]]
#    target_data = output_preprocess(act[:, 1:])
        input_data = []
        #breakpoint()
        #breakpoint()
        for arg in args[0]:
            input_data.append(arg[:, :-1])
        target_data = output_preprocess(args[0][0][:, 1:])
        with tf.GradientTape() as tape:
            logits = model(input_data, training=True) 
            loss_value = loss_function(target_data, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_loss(loss_value)
        train_accuracy(accuracy_function(target_data, logits))

    @tf.function(input_signature=[train_step_signature])
    def vali_step(*args):
#def vali_step(act, res, var):
        input_data = []
        for arg in args[0]:
            input_data.append(arg[:, :-1])
        #input_data = [act[:, :-1], res[:, :-1], var[:, :-1]]
        target_data = output_preprocess(args[0][0][:, 1:])
        logits = model(input_data, training=False) 
        loss_value = loss_function(target_data, logits)

        vali_loss(loss_value)
        vali_accuracy(accuracy_function(target_data, logits))

    bar_epoch = tqdm(range(epochs), desc='Train', position=0)
    bar_batch = tqdm(padded_ds, desc='Epoch', position=1, leave=False)
    for epoch in tqdm(range(epochs), desc='Train', position=0):
        for step, batch_data in enumerate(tqdm(padded_ds, desc='Epoch', position=1, leave=False)):
            # extract data
            input_data = []
            for feature in features:
                input_data.append(batch_data[feature['name']])
            train_step(input_data)
#            train_step([batch_data['activity'], batch_data['resource'], 
#                        batch_data['variant']])
            
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for step, batch_data in enumerate(tqdm(padded_ds_vali, desc='Vali', position=1, leave=False)):
            input_data = []
            for feature in features:
                input_data.append(batch_data[feature['name']])
            vali_step(input_data)
        with vali_summary_writer.as_default():
            tf.summary.scalar('loss', vali_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', vali_accuracy.result(), step=epoch)

        wait += 1
        if vali_loss.result() < best or epoch == 0:
            best = vali_loss.result()
            wait = 0

        train_loss.reset_states()
        train_accuracy.reset_states()
        vali_loss.reset_states()
        vali_accuracy.reset_states()

        if wait >= patience:
            break
    
    model.save(model_path)

# compute ensamble accuracy
vali_accuracy_ensamble = tf.keras.metrics.Mean(name='vali_accuracy_ensamble')
models_names = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
models = []
for model_name in models_names:
    model_path = os.path.join(model_dir, model_name)
    model = tf.keras.models.load_model(model_path)
    models.append(model)
for step, batch_data in enumerate(tqdm(padded_ds_vali, desc='Vali', position=0, leave=False)):
    input_data = []
    for feature in features:
        input_data.append(batch_data[feature['name']])
    for n, model in enumerate(models):
        @tf.function(input_signature=[train_step_signature])
        def vali_step_ensamble(*args):
#def vali_step(act, res, var):
            input_data = []
            for arg in args[0]:
                input_data.append(arg[:, :-1])
            #input_data = [act[:, :-1], res[:, :-1], var[:, :-1]]
            target_data = output_preprocess(args[0][0][:, 1:])
            logits = model(input_data, training=False) 
            return logits, target_data
        #breakpoint()
        if n == 0:
            logits_total, target_data = vali_step_ensamble(input_data)
        else:
            logits_single, target_data = vali_step_ensamble(input_data)
            logits_total += logits_single

    logits_total /= n

    vali_accuracy_ensamble(accuracy_function(target_data, logits_total))

print(vali_accuracy_ensamble.result())
