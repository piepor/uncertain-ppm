import tensorflow as tf
import yaml
import re
import numpy as np
import tensorflow_datasets as tfds
from helpdesk_utils import helpdesk_utils
from bpic2012_utils import bpic2012_utils
import os

def compute_features(file_path, vocabularies):
    with open(file_path, 'r') as file:
        features = list(yaml.load_all(file, Loader=yaml.FullLoader))
    for feature in features:
        if feature['feature-type'] == 'string':
            feature['vocabulary'] = vocabularies[feature['name']]
    return features

def loss_function(real, pred, loss_object):
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

def combine_two_string(string1, string2):
    return "{}-{}".format(string1, string2)

def reliability_diagram(target, predicted_probs, rel_dict, bins=0.05):
    #breakpoint()
    #out_dict = dict()
    count_bin = 0
    for num_bin in np.arange(0, 1, bins):
        masked = np.ma.masked_outside(np.max(predicted_probs, axis=2), num_bin, num_bin+bins) 
        acc = np.equal(target, np.argmax(predicted_probs, axis=2))
        acc = np.ma.masked_array(acc, masked.mask, fill_value=-1)
        acc = acc.mean().data.tolist()
        #out_set.add(((num_bin, num_bin+bins), acc))
        count_bin += 1
        if 'bin{}_{:.2f}_{:.2f}'.format(count_bin, num_bin, num_bin + bins) in rel_dict.keys():
            rel_dict['bin{}_{:.2f}_{:.2f}'.format(count_bin, num_bin, num_bin + bins)].append(acc)
        else:
            rel_dict['bin{}_{:.2f}_{:.2f}'.format(count_bin, num_bin, num_bin + bins)] = [acc]
        #    out_dict['bin{}'.format(count_bin)].append(rel_dict['bin{}'.format(count_bin)]
        #    out_dict['bin{}'.format(count_bin)] /= 2
    return rel_dict

def idx_to_int(tfds_id: str, builder):
    """Format the tfds_id in a more human-readable."""
    match = re.match(r'\w+-(\w+).\w+-(\d+)-of-\d+__(\d+)', tfds_id)
    split_name, shard_id, ex_id = match.groups()
    return int(ex_id)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies, axis=-1)/tf.reduce_sum(mask, axis=-1)

def single_accuracies(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    accuracies = tf.where(mask, accuracies, tf.ones(tf.shape(accuracies))*-1)
    return accuracies

def get_targets_probability(real, pred):
    one_hot = tf.one_hot(real, pred.shape[2])
    mult = pred * one_hot
    target_prob = tf.reshape(mult[mult>0], (real.shape[0], real.shape[1]))
    return target_prob
    
def compute_features(file_path, vocabularies):
    with open(file_path, 'r') as file:
        features = list(yaml.load_all(file, Loader=yaml.FullLoader))
    for feature in features:
        if feature['feature-type'] == 'string':
            feature['vocabulary'] = vocabularies[feature['name']]
    return features

def binary_crossentropy(prob):
    prob2 = 1 - prob
    entropy = prob*np.log(prob) + prob2*np.log(prob2)
    return -entropy

def max_multiclass_crossentropy(prob, num_class):
    prob2 = (1 - prob) / (num_class - 1)
    entropy = prob*np.log(prob) + (num_class - 1)*prob2*np.log(prob2)
    return -entropy

def import_dataset(dataset, ds_type, tfds_id, batch_size):
    read_config = tfds.ReadConfig()
    read_config.add_tfds_id = True

    builder_ds = tfds.builder(dataset)
    ds_train = builder_ds.as_dataset(read_config=read_config, split='train[:70%]')
    ds_vali = builder_ds.as_dataset(read_config=read_config, split='train[70%:85%]')
    ds_test = builder_ds.as_dataset(read_config=read_config, split='train[85%:]')
    if dataset == 'helpdesk':
        padded_shapes, padding_values, vocabulary_act = helpdesk_utils(tfds_id)
    elif dataset == 'bpic2012':
        padded_shapes, padding_values, vocabulary_act, vocabulary_res = bpic2012_utils(tfds_id)
    padded_ds_train = ds_train.padded_batch(batch_size, 
            padded_shapes=padded_shapes,
            padding_values=padding_values).prefetch(tf.data.AUTOTUNE)
    train_examples = len(ds_train)
    padded_ds_vali = ds_vali.padded_batch(batch_size, 
            padded_shapes=padded_shapes,
            padding_values=padding_values).prefetch(tf.data.AUTOTUNE)
    vali_examples = len(ds_vali)
    padded_ds_test = ds_test.padded_batch(batch_size, 
            padded_shapes=padded_shapes,
            padding_values=padding_values).prefetch(tf.data.AUTOTUNE)
    test_examples = len(ds_test)

    if ds_type == 'all':
        datasets = [(padded_ds_train, train_examples, 'training set'),
                (padded_ds_vali, vali_examples, 'validation set'),
                (padded_ds_test, test_examples, 'test set')]
    elif ds_type == 'training':
        datasets = [(padded_ds_train, train_examples, 'training set')]
    elif ds_type == 'validation':
        datasets = [(padded_ds_vali, vali_examples, 'validation set')]
    elif ds_type == 'test':
        datasets = [(padded_ds_test, test_examples, 'test set')]
    else:
        raise ValueError('Dataset type not understood')
    return datasets, builder_ds

def load_models(model_dir, dataset, tfds_id, model_type):
    if dataset == 'helpdesk':
        padded_shapes, padding_values, vocabulary_act = helpdesk_utils(tfds_id)
        features = compute_features(os.path.join(model_dir, 'features.params'), {'activity': vocabulary_act})
        output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary_act,
                                                         num_oov_indices=1)
    elif dataset == 'bpic2012':
        padded_shapes, padding_values, vocabulary_act, vocabulary_res = bpic2012_utils(tfds_id)
        features = compute_features(os.path.join(model_dir, 'features.params'), 
                                    {'activity': vocabulary_act, 'resource': vocabulary_res})
        output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary_act,
                                                         num_oov_indices=1)
    models_names = [name for name in os.listdir(model_dir) if os.path.isdir(
        os.path.join(model_dir, name)) and not name == 'calibrated']
    if len(models_names) == 0:
        raise OSError('No models are contained in {}'.format(model_dir))

    models = []
    for num, model_name in enumerate(models_names):
        if (model_type == 'ensamble') or (model_type == 'MC-dropout' and num == 0):
            model_path = os.path.join(model_dir, model_name)
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            
    return features, output_preprocess, models, vocabulary_act

def process_args(parser):
    args = parser.parse_args()
    dataset = args.dataset
    model_dir = os.path.join('models_ensamble', args.dataset, args.model_directory)
    if not os.path.isdir(model_dir):
        raise OSError('Directory {} does not exist'.format(model_dir)) 

    plot_entire_seqs = False
    if args.plot_entire_sequences > 0:
        plot_entire_seqs = True
    plot_wrong_preds = False
    if args.plot_wrong_predictions > 0:
        plot_wrong_preds = True
    if args.model_type == 'MC-dropout':
        dropout = True
    elif args.model_type == 'ensamble':
        dropout = False
    else:
        raise ValueError('Model type not implemented')
    if args.samples_number < 0:
        raise ValueError('Number of samples must be positive')
# thresholds
    if args.save_cases_threshold:
        try:
            os.remove('saved_cases.txt')
        except:
            pass
    return plot_entire_seqs, plot_wrong_preds, dropout

def compute_distributions(models, model_type, input_data, dropout, num_samples):
    u_a = 0
    if model_type == 'MC-dropout':
        for i in range(num_samples):
            out_prob = tf.nn.softmax(models[0](input_data, training=dropout))
            if  i == 0:
                out_prob_tot_distr = out_prob
            else:
                out_prob_tot_distr += out_prob
            # compute aleatoric uncertainty
            u_a += np.sum(out_prob.numpy()*np.log(out_prob.numpy()), axis=-1)

        out_prob_tot_distr /= num_samples
    else:
        for i, model in enumerate(models):
            out_prob = tf.nn.softmax(model(input_data, training=dropout))
            if  i == 0:
                out_prob_tot_distr = out_prob
            else:
                out_prob_tot_distr += out_prob
            # compute aleatoric uncertainty
            u_a += np.sum(out_prob.numpy()*np.log(out_prob.numpy()), axis=-1)
        out_prob_tot_distr /= len(models)
    return u_a, out_prob_tot_distr, out_prob
