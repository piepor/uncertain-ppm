import tensorflow as tf
import yaml
import re
import numpy as np
import tensorflow_datasets as tfds
from helpdesk_utils import helpdesk_utils
from bpic2012_utils import bpic2012_utils
import os
from pm4py.objects.log.util import interval_lifecycle
from pm4py.util import constants
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log.obj import EventLog
from scipy.stats import percentileofscore
import pm4py

def accuracy_top_k(target_data, output_matrix, k):
    indx = np.argsort(output_matrix)
    indx_top_k = indx[:, :, -k:] 
    acc_no_mask = np.equal(indx_top_k, target_data[:, :, np.newaxis]).any(axis=2)
    masked = np.ma.masked_equal(target_data, 0).mask
    acc_masked = np.ma.masked_array(acc_no_mask, mask=masked)
    return acc_masked.mean()

def predict_case(models, case_id, data, builder_ds, dropout, num_samples, features, model_type):
    for batch_idx, batch_data in enumerate(data):
        #exs = [ex.numpy().decode('utf-8') for ex in batch_data['tfds_id']]
        #target_data_case = [idx_to_int(tfds_id, builder_ds) for tfds_id in exs] 
        target_data_case = [ex.numpy()[0] for ex in batch['case_id']]
        breakpoint()
        if int(case_id) in target_data_case:
            breakpoint()
            index = np.where(np.asarray(target_data_case) == int(case_id))
            index = index[0][0]
            input_data = []
            for feature in features:
                input_data.append(tf.expand_dims(batch_data[feature['name']][index, :-1], axis=0))
            target_data = batch_data['activity'][index, 1:]
            u_a, out_ensamble, out_single = compute_distributions(
                models, model_type, input_data, dropout, num_samples)
    return out_ensamble, out_single, target_data

def extract_case(case_id, log, dataset):
    case = None
    for trace in log:
        if dataset == 'helpdesk':
            if trace[0]['Case ID'].split()[1] == str(case_id):
                case = trace
        elif dataset == 'bpic2012':
            if trace.attributes['concept:name'] == str(case_id):
                case = trace
        else:
            raise Exception('Wrong dataset name')
    if case is None:
        raise Exception('Case not found')
    return case

def get_variant_characteristics(filtered_log, features):
    #breakpoint()
    features_dict = {key: list() for key in features.keys()}
    features_dict['day_part'] = list()
    features_dict['week_day'] = list()
    features_dict['completion_time'] = list()
    for trace in filtered_log:
        for count, event in enumerate(trace):
            for feature in features.keys():
                if features[feature] == 'event':
                    features_dict[feature].append(event[feature])
                elif features[feature] == 'trace' and count == 0:
                    features_dict[feature].append(trace.attributes[feature])
            features_dict['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
            features_dict['week_day'].append(event['time:timestamp'].isoweekday())
        features_dict['completion_time'].append(trace[-1]['time:timestamp'] - trace[0]['time:timestamp'])
    return features_dict

def get_case_characteristic(case, features):
    case_dict = dict()
    for feature in features:
        case_dict[feature] = list()
    case_dict['day_part'] = list()
    case_dict['week_day'] = list()
    for event in case:
        for feature in features:
            case_dict[feature].append(event[feature])
        case_dict['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
        case_dict['week_day'].append(event['time:timestamp'].isoweekday())
    return case_dict

def get_case_statistics(case_dict, features_dict, features_type, case_seq):
    case_stats = dict()
    for feature in case_dict:
        unique_values = set(case_dict[feature])
        case_stats[feature] = dict.fromkeys(unique_values)
        if features_type[feature] == 'categorical':
            for unique_value in unique_values:
                perc = case_dict[feature].count(unique_value) / features_dict[feature].count(unique_value)
                case_stats[feature][unique_value] = perc
        elif features_type[feature] == 'continuous':
            perc = percentileofscore(features_dict[feature], unique_value)
    seconds_completion_time = [time_data.seconds + time_data.days*24*3600 for time_data in features_dict['completion_time']]
    duration_case = case_seq[-1]['time:timestamp'] - case_seq[0]['time:timestamp']
    duration_case = duration_case.seconds + duration_case.days*24*3600
    case_stats['completion_time'] = percentileofscore(seconds_completion_time, duration_case)
    return case_stats

def get_variant_from_case(case):
    variant = ''
    for event in case:
        variant += ',{}'.format(event['concept:name'])
    return variant[1:]

def filter_variant(log, variant):
    filtered_log = EventLog(list(), attributes=log.attributes, extensions=log.extensions, 
                            classifiers=log.classifiers, omni_present=log.omni_present, properties=log.properties)
    for trace in log:
        case_variant = get_variant_from_case(trace) 
        if case_variant == variant:
            filtered_log.append(trace)
    return filtered_log

def get_variants_percentage(log):
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
    tot_sum = 0
    for variant in variants_count:
        tot_sum += variant['count']
    variants_count_perc = []
    for variant in variants_count:
        variants_count_perc.append({'variant': variant['variant'],
                                    'count': variant['count'],
                                    'perc': variant['count']/tot_sum})
    return variants_count_perc, variants_count

def get_case_percentage(log, case_id, variants):
    #filtered_log = pm4py.filter_log(lambda x: x.attributes['concept:name'] == str(case_id), log) 
    #breakpoint()
    filtered_log = pm4py.filter_log(lambda x: x[0]['Case ID'].split()[1] == str(case_id), log) 
    case_events = ''
    for event in filtered_log[0]:
        case_events = case_events + ',' + event['concept:name']
    case_events = case_events[1:]
    case_perc = list(filter(lambda x: x['variant']==case_events, variants))
    case_perc = case_perc[0]['perc']
    return case_events, case_perc

def compute_bin_data(u_t_array_single, acc_array_single):
    bin_size_perc = 0.15
    max_unc = np.max(u_t_array_single)
    num_bins = int(np.ceil(max_unc / bin_size_perc))
    perc_right_plot = []
    perc_wrong_plot = []
    u_t_plot = []
    acc_plot = []
    perc_data = []
    for count_bin in np.arange(0, max_unc, bin_size_perc):
        tot_pred = u_t_array_single[
            (u_t_array_single >= count_bin) & (u_t_array_single < count_bin+bin_size_perc)]
        acc_pred = acc_array_single[
            (u_t_array_single >= count_bin) & (u_t_array_single < count_bin+bin_size_perc)]
        acc_plot.append(np.mean(acc_pred))
        perc_right = len(tot_pred[acc_pred == 1]) / len(tot_pred)
        perc_wrong = 1 - perc_right
        perc_data_tot = len(tot_pred) / len(u_t_array_single)
        perc_data.append(perc_data_tot)
        perc_right_plot.append(perc_right*perc_data_tot)
        perc_wrong_plot.append(perc_wrong*perc_data_tot)
        u_t_plot.append(bin_size_perc)
    return u_t_plot, perc_right_plot, perc_wrong_plot, perc_data, acc_plot

def expected_calibration_error(rel_dict):
    ece = 0
    num_total_valid_data = 0
    #breakpoint()
    for key in rel_dict.keys(): 
        num_total_valid_data += rel_dict[key][:, 2].sum() 
    #breakpoint()
    for key in rel_dict.keys(): 
        #breakpoint()
        acc = rel_dict[key][:, 0 ].mean()
        conf = rel_dict[key][:, 1 ].mean()
        perc_data = rel_dict[key][:, 2].sum() / num_total_valid_data
        ece += perc_data * np.abs(acc - conf) 
    return ece

def anomaly_detection_isolation_forest(log):
    log = interval_lifecycle.assign_lead_cycle_time(log, parameters={
        constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY: "start_timestamp",
        constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"
    })
    data, features_names = log_to_features.apply(log, 
                                                 parameters={"str_ev_attr": ["concept:name", "org:resource"],
                                                             "str_tr_attr": [], "num_ev_attr": ["@@approx_bh_partial_cycle_time"],
                                                             "num_tr_attr": [], "str_evsucc_attr": ["concept:name", "org:resource"]})
    df = pd.DataFrame(data, columns=features_names)
    pca = PCA(n_components=5)
    df2 = pd.DataFrame(pca.fit_transform(df))
    model = IsolationForest()
    model.fit(df2)
    df2["scores"] = model.decision_function(df2)
    df2["@@index"] = df2.index
    df2 = df2.sort_values("scores")
    return df2.loc[df2["scores"]<0].index.to_numpy()

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
    #breakpoint()
    for num_bin in np.arange(0, 1, bins):
        #breakpoint()
        masked_pad = np.ma.masked_equal(target, 0) 
        masked = np.ma.masked_outside(np.max(predicted_probs, axis=2), num_bin, num_bin+bins) 
        valid_bin_mask = masked.mask | masked_pad.mask
        acc = np.equal(target, np.argmax(predicted_probs, axis=2))
        conf = np.max(predicted_probs, axis=2)
        acc = np.ma.array(acc.tolist(), mask=valid_bin_mask.tolist())
        conf = np.ma.array(conf.tolist(), mask=valid_bin_mask.tolist())
        num_data = len(acc.compressed())
        num_valid_pad_data = len(masked_pad.compressed())
        #acc = np.ma.masked_array(acc, total_mask, fill_value=-1)
        acc = acc.mean().data.tolist()
        conf = conf.mean().data.tolist()
        #out_set.add(((num_bin, num_bin+bins), acc))
        count_bin += 1
        key_dict = 'bin{}_{:.2f}_{:.2f}'.format(count_bin, num_bin, num_bin + bins) 
        #breakpoint()
        if key_dict in rel_dict.keys():
            rel_dict[key_dict] = np.vstack((rel_dict[key_dict], [acc, conf, num_data]))
            #rel_dict['bin{}_{:.2f}_{:.2f}'.format(count_bin, num_bin, num_bin + bins)].append((acc, conf, len(acc.compressed())
        else:
            rel_dict[key_dict] = np.asarray([acc, conf, num_data])
            #rel_dict['bin{}_{:.2f}_{:.2f}'.format(count_bin, num_bin, num_bin + bins)] = [acc]
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
        padded_shapes, padding_values, vocabulary_act, features_type, features_variant = helpdesk_utils(tfds_id)
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
        padded_shapes, padding_values, vocabulary_act, features_type, features_variant = helpdesk_utils(tfds_id)
        features = compute_features(os.path.join(model_dir, 'features.params'), {'activity': vocabulary_act})
        output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary_act,
                                                         num_oov_indices=1)
    elif dataset == 'bpic2012':
        padded_shapes, padding_values, vocabulary_act, vocabulary_res, features_type, features_variant = bpic2012_utils(tfds_id)
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
            
    return features, output_preprocess, models, vocabulary_act, features_type, features_variant

def process_args(parser):
    args = parser.parse_args()
    dataset = args.dataset
    model_dir = os.path.join('models', args.dataset, args.model_directory)
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
