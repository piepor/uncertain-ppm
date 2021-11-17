import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
import datetime
import glob
from shutil import copyfile
import argparse

from utils import compute_features, loss_function, accuracy_function, compute_input_signature
from model_utils import GeneralModel
from helpdesk_utils import helpdesk_utils
from bpic2012_utils import bpic2012_utils

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='training dataset', 
                    choices=['helpdesk', 'bpic2012'])
parser.add_argument('features', 
                    help='yaml file specifying training features (e.g. act_res_time.params)')
parser.add_argument('--num_heads', help='number of heads in multi-head attention',
                    default=2, type=int)
parser.add_argument('--feed_forward_dim', help='units in feed forward layers',
                    default=512, type=int)
parser.add_argument('--epochs', help='number of training epochs',
                    default=100, type=int)
parser.add_argument('--patience', help='epochs of patience before stopping the training',
                    default=10, type=int)
parser.add_argument('--batch_size', help='size of training batches',
                    default=32, type=int)
parser.add_argument('--ensamble_number', help='number of models in the ensamble',
                    default=5, type=int)

args = parser.parse_args()
dataset = args.dataset
features_name = os.path.join('models_features', dataset, args.features)
num_heads = args.num_heads
feed_forward_dim = args.feed_forward_dim
epochs = args.epochs
patience = args.patience
batch_size = args.batch_size
num_models_ensamble = args.ensamble_number

if dataset == 'helpdesk':
    ds_train = tfds.load(dataset, split='train[:70%]', shuffle_files=True)
    ds_vali = tfds.load(dataset, split='train[70%:85%]')
    ds_test = tfds.load(dataset, split='train[85%:]')
    padded_shapes, padding_values, vocabulary = helpdesk_utils()
    features = compute_features(features_name, {'activity': vocabulary})
    output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary,
                                                     num_oov_indices=1)
elif dataset == 'bpic2012':
    ds_train = tfds.load(dataset, split='train[:70%]', shuffle_files=True)
    ds_vali = tfds.load(dataset, split='train[70%:85%]')
    ds_test = tfds.load(dataset, split='train[85%:]')
    padded_shapes, padding_values, vocabulary_act, vocabulary_res = bpic2012_utils()
    features = compute_features(features_name, 
                                {'activity': vocabulary_act, 'resource': vocabulary_res})
    output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary_act,
                                                     num_oov_indices=1)

train_step_signature = compute_input_signature(features)
embed_dim = sum([feature['output-dim'] for feature in features])
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam()
    
ensamble_list = glob.glob('models_ensamble/{}/ensamble*'.format(dataset))
num = 0
for ensamble in ensamble_list:
    if int(ensamble.split('_')[2]) > num:
        num = int(ensamble.split('_')[2])
model_dir = 'models_ensamble/{}/ensamble_{}'.format(dataset, int(num)+1)
os.makedirs(model_dir)
copyfile(features_name, os.path.join(model_dir, 'features.params'))

# train loop
wait = 0
best = 0

padded_ds = ds_train.padded_batch(batch_size, 
        padded_shapes=padded_shapes,
        padding_values=padding_values).shuffle(buffer_size=10000, 
                                               reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
padded_ds_vali = ds_vali.padded_batch(batch_size, 
                                      padded_shapes=padded_shapes,
                                      padding_values=padding_values).prefetch(tf.data.AUTOTUNE)

for _ in range(num_models_ensamble):
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
        input_data = []
        for arg in args[0]:
            input_data.append(arg[:, :-1])
        target_data = output_preprocess(args[0][0][:, 1:])
        with tf.GradientTape() as tape:
            logits = model(input_data, training=True) 
            loss_value = loss_function(target_data, logits, loss_object)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_loss(loss_value)
        train_accuracy(accuracy_function(target_data, logits))

    @tf.function(input_signature=[train_step_signature])
    def vali_step(*args):
        input_data = []
        for arg in args[0]:
            input_data.append(arg[:, :-1])
        target_data = output_preprocess(args[0][0][:, 1:])
        logits = model(input_data, training=False) 
        loss_value = loss_function(target_data, logits, loss_object)

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
            input_data = []
            for arg in args[0]:
                input_data.append(arg[:, :-1])
            target_data = output_preprocess(args[0][0][:, 1:])
            logits = model(input_data, training=False) 
            return logits, target_data
        if n == 0:
            logits_total, target_data = vali_step_ensamble(input_data)
        else:
            logits_single, target_data = vali_step_ensamble(input_data)
            logits_total += logits_single

    logits_total /= n

    vali_accuracy_ensamble(accuracy_function(target_data, logits_total))

print(vali_accuracy_ensamble.result())
