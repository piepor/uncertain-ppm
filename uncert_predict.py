import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import yaml
import tensorflow_datasets as tfds
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import argparse
import re
from utils import compute_features, loss_function, accuracy_function, compute_input_signature
from helpdesk_utils import helpdesk_utils
from bpic2012_utils import bpic2012_utils

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

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='choose the dataset',
                    choices=['helpdesk', 'bpic2012'])
parser.add_argument('model_directory', help='directory where the ensamble models are saved')
parser.add_argument('--plot_entire_sequences', help='plot the output distribution of N random sequences',
        default=0, type=int)
parser.add_argument('--plot_wrong_predictions', help='plot N output distribution of wrong predictions',
        default=0, type=int)
parser.add_argument('--dataset_type', help='choose what segment of dataset to use between training, validation and test. Default is all',
        default='all', choices=['training', 'validation', 'test', 'all'])
parser.add_argument('--model_type', help='choose the type of algorithm used to estimate uncertainty: MC dropout or ensamble. Default to ensamble',
        default='ensamble', choices=['MC-dropout', 'ensamble'])
parser.add_argument('--samples_number', help='number of sample for the MC dropout',
        default=5, type=int)
parser.add_argument('--batch_size', help='size of batches',
        default=64, type=int)
parser.add_argument('--uncertainty_threshold', help='uncertainty threshold to select cases', 
        default=0.4, type=float)
parser.add_argument('--plot_cases_threshold', help='plot cases below the threshold', 
        default=False, type=bool)
parser.add_argument('--save_cases_threshold', help='save cases below the threshold', 
        default=False, type=bool)
parser.add_argument('--plot_mean_accuracy_vs_uncertainty', help='plot mean seqeunces accuracy vs uncertainty', 
        default=False, type=bool)
parser.add_argument('--plot_event_probability_vs_uncertainty', help='plot single event accuracy vs uncertainty', 
        default=True, type=bool)
parser.add_argument('--plot_event_correctness_vs_uncertainty', help='plot single event correctness vs uncertainty', 
        default=False, type=bool)
parser.add_argument('--plot_box_plot_uncertainty', help='plot box plot of uncertainties', 
        default=True, type=bool)
parser.add_argument('--plot_distributions', help='plot distribution of uncertainties', 
        default=False, type=bool)
parser.add_argument('--plot_reliability_diagram', help='plot reliability diagram', 
        default=True, type=bool)
parser.add_argument('--plot_accuracy_vs_uncertainty', help='box plot of accuracy vs uncertainty', 
        default=True, type=bool)
parser.add_argument('--plot_proportions', 
                    help='box plot of right and wrong predictions normalize with the percentage of data in bin.', 
        default=False, type=bool)
parser.add_argument('--tfds_id', help='extract also the case id from dataset', 
        default=True, type=bool)

# Parse and check arguments
args = parser.parse_args()

dataset = args.dataset.lower()
model_dir = os.path.join('models_ensamble', dataset.lower(), args.model_directory)
model_number = args.model_directory.split('_')[1]
if not os.path.isdir(model_dir):
    raise OSError('Directory {} does not exist'.format(model_dir)) 

plot_entire_seqs = False
num_seq_entire = args.plot_entire_sequences
if num_seq_entire > 0:
    plot_entire_seqs = True
plot_wrong_preds = False
num_wrong_preds = args.plot_wrong_predictions
if num_wrong_preds > 0:
    plot_wrong_preds = True
ds_type = args.dataset_type.lower()
#if not dataset in ['helpdesk']:
#    raise ValueError('Dataset not available') 
model_type = args.model_type
if model_type == 'MC-dropout':
    dropout = True
elif model_type == 'ensamble':
    dropout = False
else:
    raise ValueError('Model type not implemented')
num_samples = args.samples_number
if num_samples < 0:
    raise ValueError('Number of samples must be positive')

# thresholds
unc_threshold = args.uncertainty_threshold
plot_threshold = args.plot_cases_threshold
save_threshold = args.save_cases_threshold
if save_threshold:
    try:
        os.remove('saved_cases.txt')
    except:
        pass
batch_size = args.batch_size
acc_threshold = 0.5

# plot stats
plot_mean_acc = args.plot_mean_accuracy_vs_uncertainty
plot_event_prob = args.plot_event_probability_vs_uncertainty
plot_event_corr = args.plot_event_correctness_vs_uncertainty
plot_box_plot = args.plot_box_plot_uncertainty
plot_distr = args.plot_distributions
plot_reliability_diagram = args.plot_reliability_diagram
plot_acc_unc = args.plot_accuracy_vs_uncertainty
plot_acc_unc = args.plot_accuracy_vs_uncertainty
plot_proportions = args.plot_proportions
tfds_id = args.tfds_id

# Start analysis - import dataset
# change read config in order to return also the case id
read_config = tfds.ReadConfig()
read_config.add_tfds_id = True

builder_ds = tfds.builder(dataset)
ds_train = builder_ds.as_dataset(read_config=read_config, split='train[:70%]')
ds_vali = builder_ds.as_dataset(read_config=read_config, split='train[70%:85%]')
ds_test = builder_ds.as_dataset(read_config=read_config, split='train[85%:]')
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
#batch_size = 64
#breakpoint()
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

models_names = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
if len(models_names) == 0:
    raise OSError('No models are contained in {}'.format(model_dir))

models = []
for num, model_name in enumerate(models_names):
    if (model_type == 'ensamble') or (model_type == 'MC-dropout' and num == 0):
        model_path = os.path.join(model_dir, model_name)
        model = tf.keras.models.load_model(model_path)
        models.append(model)

count_seq = 0
count_wrong = 0
for ds, num_examples, ds_name in datasets:
    if plot_entire_seqs or plot_wrong_preds:
        count_seq = 0
        count_wrong = 0
        random_idx_plot = random.sample(range(num_examples), num_seq_entire)

    u_t_array_mean = np.zeros(1)
    u_t_array_single = np.zeros(1)
    u_a_array_mean = np.zeros(1)
    u_a_array_single = np.zeros(1)
    u_e_array_mean = np.zeros(1)
    u_e_array_single = np.zeros(1)
    acc_array_mean = np.zeros(1)
    acc_array_single = np.zeros(1)
    target_prob_array = np.zeros(1)
    max_prob_event_array = np.zeros(1)
    target_label = np.asarray(['0'])
    target_case = np.asarray(['0'])
    rel_dict = dict()
    rel_dict_one_model = dict()
    for batch_idx, batch_data in enumerate(ds):

        input_data = []
        for feature in features:
            input_data.append(batch_data[feature['name']][:, :-1])
        target_data = batch_data['activity'][:, 1:]
        #breakpoint()
        exs = [ex.numpy().decode('utf-8') for ex in batch_data['tfds_id']]
        target_data_case = [idx_to_int(tfds_id, builder_ds) for tfds_id in exs] 
        target_data_case = target_data_case * np.ones_like(target_data).T
        target_data_case = target_data_case.T
        target_label = np.hstack([target_label, target_data.numpy().ravel()]) 
        #breakpoint()
        target_case = np.hstack([target_case, target_data_case.ravel()]) 
        target_data = output_preprocess(target_data)
        mask = tf.math.logical_not(tf.math.equal(target_data, 0))

        u_a = 0
        if model_type == 'MC-dropout':
            for i in range(num_samples):
                out_prob = tf.nn.softmax(model(input_data, training=dropout))
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

        rel_bins = 0.05
        rel_dict = reliability_diagram(target_data.numpy(), out_prob_tot_distr.numpy(), rel_dict, rel_bins)
        rel_dict_one_model = reliability_diagram(target_data.numpy(), out_prob.numpy(), rel_dict_one_model, rel_bins)
        #breakpoint()
        acc = accuracy_function(target_data, out_prob_tot_distr)
        acc_array_mean = np.hstack([acc_array_mean, acc])
        max_prob_event = tf.reduce_max(out_prob_tot_distr, axis=2)
        max_prob_event_array = np.hstack([max_prob_event_array, max_prob_event.numpy().ravel()])

        acc_single = single_accuracies(target_data, out_prob_tot_distr)
        acc_array_single = np.hstack([acc_array_single, acc_single.numpy().ravel()])
        
        target_prob = get_targets_probability(target_data, out_prob_tot_distr)
        target_prob_array = np.hstack([target_prob_array, target_prob.numpy().ravel()])

        if model_type == 'MC-dropout':
            u_a /= -num_samples
        else:
            u_a /= -len(models)
        u_a = np.where(mask, u_a, np.zeros(np.shape(u_a)))
        u_a_array_single = np.hstack([u_a_array_single, u_a.ravel()])
        # compute total uncertainty
        u_t = -np.sum(out_prob_tot_distr.numpy() * np.log(out_prob_tot_distr.numpy()), axis=-1)
        u_t = np.where(mask, u_t, np.zeros(np.shape(u_t)))
        u_t_array_single = np.hstack([u_t_array_single, u_t.ravel()])
        # compute epistemic uncertainty
        u_e = u_t - u_a
        u_e_array_single = np.hstack([u_e_array_single, u_e.ravel()])

        vocabulary_plot = ['<PAD>']
        vocabulary_plot.extend(vocabulary_act)

        length_seq = tf.reduce_sum(
                tf.cast(tf.math.logical_not(
                    tf.math.equal(target_data, 0)), dtype=tf.float32),
                axis=-1)
        mean_u_t = np.sum(u_t, axis=-1) / length_seq.numpy()
        mean_u_a = np.sum(u_a, axis=-1) / length_seq.numpy()
        mean_u_e = np.sum(u_e, axis=-1) / length_seq.numpy()

        u_t_array_mean = np.hstack([u_t_array_mean, mean_u_t])
        u_a_array_mean = np.hstack([u_a_array_mean, mean_u_a])
        u_e_array_mean = np.hstack([u_e_array_mean, mean_u_e])

        check_cond_tot_unc_prob = np.logical_and(u_t<unc_threshold, mask).any() \
            and np.logical_and(acc_single<acc_threshold, mask).any()
        check_cond_wrong = np.logical_and(acc_single==0.0, mask).any() \
            and plot_wrong_preds and count_wrong<num_wrong_preds
        check_cond_entire = plot_entire_seqs and count_seq<num_seq_entire
        # mask the row if an event of the case respect the condition 
        prob_unc_mask = np.logical_and(tf.math.less(acc_single, acc_threshold).numpy(), u_t<unc_threshold)
        prob_unc_mask = np.logical_and(prob_unc_mask, mask)
        prob_unc_mask = np.ma.masked_equal(prob_unc_mask, True)
        prob_unc_mask = np.ma.mask_rows(prob_unc_mask).mask
        if check_cond_tot_unc_prob and save_threshold:
            case_names = target_case[1:]
            #breakpoint()
            case_selected = target_data_case[prob_unc_mask]
            case_selected = case_selected.reshape((-1, prob_unc_mask.shape[1]))
            for num_row in range(case_selected.shape[0]):
                with open('saved_cases.txt', 'a') as file:
                    file.write("{}\n".format(case_selected[num_row, 0]))
            #breakpoint()

        if check_cond_wrong or check_cond_entire or (check_cond_tot_unc_prob and (save_threshold or plot_threshold)):
            # mask the row if an event of the case respect the condition 
#            prob_unc_mask = np.logical_and(tf.math.less(acc_single, acc_threshold).numpy(), u_t<unc_threshold)
#            prob_unc_mask = np.logical_and(prob_unc_mask, mask)
#            prob_unc_mask = np.ma.masked_equal(prob_unc_mask, True)
#            prob_unc_mask = np.ma.mask_rows(prob_unc_mask).mask
            if not np.ma.masked_equal(prob_unc_mask, True).mask.any():
                prob_unc_mask = np.zeros_like(acc_single)
            for num_row in range(acc_single.shape[0]):
                check_cond_wrong_row = np.logical_and(acc_single[num_row, :]==0.0, mask[num_row, :]).any() and check_cond_wrong
                if plot_wrong_preds:
                    check_cond_random_idx = (batch_idx * batch_size + num_row in random_idx_plot) 
                else:
                    check_cond_random_idx = False
                if check_cond_wrong_row or check_cond_random_idx or (prob_unc_mask[num_row, 0] and plot_threshold):
                    act = ''
                    for j in range(target_data.shape[1]):
                        act += '{} - '.format(input_data[0][num_row, j].numpy().decode("utf-8"))
                        check_cond_wrong_single_pred = check_cond_wrong_row and acc_single[num_row, j] == 0.0 and count_wrong<num_wrong_preds
                        if (check_cond_wrong_single_pred or check_cond_random_idx or (plot_threshold and prob_unc_mask[num_row, 0])) and not target_data[num_row, j].numpy() == 0:
                            target_numpy = np.zeros(len(vocabulary_plot))
                            target_numpy[target_data[num_row, j].numpy()] = 1
                            #fig = go.Figure()
                            fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2])
                            fig.add_trace(go.Bar(x=vocabulary_plot, y=target_numpy,
                                                 marker=dict(opacity=0.4), name='actual event'),
                                          row=1, col=1)
                            fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob[num_row, j].numpy(),
                                                 marker=dict(opacity=0.4), name='single model'),
                                          row=1, col=1)
                            fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob_tot_distr[num_row, j].numpy(),
                                                 marker=dict(opacity=0.4), name='{} models'.format(model_type)),
                                          row=1, col=1)
                            fig.add_trace(go.Bar(x=['Uncertainty'], y=[u_t[num_row, j]], name='Epistemic', offsetgroup=0),
                                          row=1, col=2)
                            fig.add_trace(go.Bar(x=['Uncertainty'], y=[u_a[num_row, j]], name='Aleatoric', offsetgroup=0),
                                          row=1, col=2)
                            fig.layout['yaxis2'].update(range=[0, 1.2*np.max(u_t)])
                            fig.update_layout(barmode='overlay', title_text="Model {}-{}-{}<br><sup>{}</sup>".format(
                                model_number, model_type, ds_name.capitalize(), act))
                            #fig.update_traces(opacity=0.6)
                            fig.show(renderer='chromium')
                            #breakpoint()
                            count_wrong += 1
                    count_seq += 1
#                    print("Mean total uncertainty: {}".format(mean_u_t[i]))
#                    print("Mean aleatoric uncertainty: {}".format(mean_u_a[i]))
#                    print("Mean epistemic uncertainty: {}".format(mean_u_e[i]))
                    #breakpoint()

    if plot_reliability_diagram:
        rel_bin_plot = []
        rel_acc_plot = []
        rel_acc_plot_one_model = []
        for key in rel_dict.keys():
            rel_bin_plot.append(rel_bins)
            rel_acc_plot.append(np.mean(rel_dict[key]))
            rel_acc_plot_one_model.append(np.mean(rel_dict_one_model[key]))
        #breakpoint()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=np.cumsum(rel_bin_plot)-rel_bin_plot, y=rel_acc_plot, 
                             width=rel_bin_plot, offset=0, name='ensamble', opacity=0.7))
        fig.add_trace(go.Bar(x=np.cumsum(rel_bin_plot)-rel_bin_plot, y=rel_acc_plot_one_model, 
                             width=rel_bin_plot, offset=0, name='one model', opacity=0.7))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='black', dash='dash')))
        fig.update_layout(title_text='Reliability Diagram - Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Confidence')
        fig.update_yaxes(title_text='Accuracy')
        fig.show(renderer='chromium')

    u_t_array = u_t_array_mean[1:]
    u_a_array = u_a_array_mean[1:]
    u_e_array = u_e_array_mean[1:]
    acc_array = acc_array_mean[1:]
    acc_array_single = acc_array_single[1:]
    u_t_array_single = u_t_array_single[1:]
    u_a_array_single = u_a_array_single[1:]
    u_e_array_single = u_e_array_single[1:]
    target_prob_array = target_prob_array[1:]
    target_label = target_label[1:]
    target_case = target_case[1:]
    max_prob_event_array = max_prob_event_array[1:]
    #breakpoint()
# ordering
    ordered_acc_array = acc_array_single.argsort()
    acc_array_single = acc_array_single[ordered_acc_array]
    u_t_array_single = u_t_array_single[ordered_acc_array]
    u_a_array_single = u_a_array_single[ordered_acc_array]
    u_e_array_single = u_e_array_single[ordered_acc_array]
    target_prob_array = target_prob_array[ordered_acc_array]
    target_label_array = target_label[ordered_acc_array]
    max_prob_event_array = max_prob_event_array[ordered_acc_array]
    target_case_array = target_case[ordered_acc_array]
    #breakpoint()

    if plot_distr:
        group_labels = ['Total uncertainty', 'Aleatoric uncertainty', 'Epistemic uncertainty']
        hist_data = [u_t_array_single, u_a_array_single, u_e_array_single]
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

    if plot_mean_acc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_t_array, y=acc_array, mode='markers', name='Total uncertainty'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Total uncertainty')
        fig.update_yaxes(title_text='Mean sequence accuracy')
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_a_array, y=acc_array, mode='markers', name='Aleatoric uncertainty'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_yaxes(title_text='Mean sequence accuracy')
        fig.update_xaxes(title_text='Aleatoric uncertainty')
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_e_array, y=acc_array, mode='markers', name='Epistemic uncertainty'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_yaxes(title_text='Mean sequence accuracy')
        fig.update_xaxes(title_text='Epistemic uncertainty')
        fig.show(renderer='chromium')

# select valid unceratinties and divide between right and wrong predictions
    #breakpoint()
    u_t_array_single = u_t_array_single[acc_array_single>-1]
    u_a_array_single = u_a_array_single[acc_array_single>-1]
    u_e_array_single = u_e_array_single[acc_array_single>-1]
    target_prob_array = target_prob_array[acc_array_single>-1]
    target_label_array = target_label_array[acc_array_single>-1]
    target_case_array = target_case_array[acc_array_single>-1]
    max_prob_event_array = max_prob_event_array[acc_array_single>-1]
    acc_array_single = acc_array_single[acc_array_single>-1]
    u_t_array_single_right = u_t_array_single[acc_array_single == 1]
    u_t_array_single_wrong = u_t_array_single[acc_array_single == 0]
    u_a_array_single_right = u_a_array_single[acc_array_single == 1]
    u_a_array_single_wrong = u_a_array_single[acc_array_single == 0]
    u_e_array_single_right = u_e_array_single[acc_array_single == 1]
    u_e_array_single_wrong = u_e_array_single[acc_array_single == 0]
    # Compute bins to highligth percentage of wrong and right prediction
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

    if plot_proportions:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=perc_right_plot, 
                             width=u_t_plot, offset=0, name='right predictions'))
        fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=perc_wrong_plot,
                             width=u_t_plot, offset=0, name='wrong predictions'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

    if plot_acc_unc:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=acc_plot, 
                             width=u_t_plot, offset=0, name='accuracy'))
        fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=perc_data, 
                             width=u_t_plot, offset=0, name='data percentage'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='total uncertainty')
        fig.update_yaxes(title_text='accuracy')
        fig.show(renderer='chromium')

    if plot_event_prob:
        total_label = list(map(combine_two_string, target_case_array, target_label_array))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_t_array_single_right, y=target_prob_array[acc_array_single==1],
            mode='markers', text=total_label, name='right prediction'))
        fig.add_trace(go.Scatter(x=u_t_array_single_wrong, y=target_prob_array[acc_array_single==0],
            mode='markers', text=total_label, name='wrong prediction'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Total uncertainty')
        fig.update_yaxes(title_text='Assigned probability')
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_a_array_single_right, y=target_prob_array[acc_array_single==1],
            mode='markers', text=target_label_array.tolist(), name='right prediction'))
        fig.add_trace(go.Scatter(x=u_a_array_single_wrong, y=target_prob_array[acc_array_single==0],
            mode='markers', text=target_label_array.tolist(), name='wrong prediction'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Aleatoric uncertainty')
        fig.update_yaxes(title_text='Assigned probability')
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_e_array_single_right, y=target_prob_array[acc_array_single==1],
            mode='markers', text=target_label_array.tolist(), name='right prediction'))
        fig.add_trace(go.Scatter(x=u_e_array_single_wrong, y=target_prob_array[acc_array_single==0],
            mode='markers', text=target_label_array.tolist(), name='wrong prediction'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Epistemic uncertainty')
        fig.update_yaxes(title_text='Assigned probability')
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_t_array_single_right, y=max_prob_event_array[acc_array_single==1],
            mode='markers', text=total_label, name='right prediction'))
        fig.add_trace(go.Scatter(x=u_t_array_single_wrong, y=max_prob_event_array[acc_array_single==0],
            mode='markers', text=total_label, name='wrong prediction'))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Total uncertainty')
        fig.update_yaxes(title_text='Max probability predicted')
        fig.show(renderer='chromium')

    if plot_event_corr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_t_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
            marker=dict(color=target_prob_array, colorbar=dict(title='Assigned probability'), colorscale='Viridis'),
            mode='markers', text=target_label_array.tolist()))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Total uncertainty')
        fig.update_yaxes(title_text='Point prediction correctness')
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_a_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
            marker=dict(color=target_prob_array, colorbar=dict(title='Assigned probability'), colorscale='Viridis'),
            mode='markers', text=target_label_array.tolist()))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Aleatoric uncertainty')
        fig.update_yaxes(title_text='Point prediction correctness')
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_e_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
            marker=dict(color=target_prob_array, colorbar=dict(title='Assigned probability'), colorscale='Viridis'),
            mode='markers', text=target_label_array.tolist()))
        fig.update_layout(title_text='Model {} - {} - {}'.format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.update_xaxes(title_text='Epistemic uncertainty')
        fig.update_yaxes(title_text='Point prediction correctness')
        fig.show(renderer='chromium')

# Plot distribution
    if plot_distr:
        group_labels = ['Right predictions', 'Wrong predictions']
        hist_data = [u_t_array_single_right, u_t_array_single_wrong]
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.05)
        fig.update_layout(title_text="Model {} - {} - {} - Total uncertainty".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

        hist_data = [u_a_array_single_right, u_a_array_single_wrong]
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.05)
        fig.update_layout(title_text="Model {} - {} - {} - Aleatoric uncertainty".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

        hist_data = [u_e_array_single_right, u_e_array_single_wrong]
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.05)
        fig.update_layout(title_text="Model {} - {} - {} - Epistemic uncertainty".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

# Boxplot
    if plot_box_plot:
        fig = go.Figure()
        fig.add_trace(go.Box(y=u_t_array_single_right, name='Right predictions'))
        fig.add_trace(go.Box(y=u_t_array_single_wrong, name='Wrong predictions'))
        fig.update_layout(title_text="Model {} - {} - {} - Total uncertainty".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Box(y=u_a_array_single_right, name='Right predictions'))
        fig.add_trace(go.Box(y=u_a_array_single_wrong, name='Wrong predictions'))
        fig.update_layout(title_text="Model {} - {} - {} - Aleatoric uncertainty".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Box(y=u_e_array_single_right, name='Right predictions'))
        fig.add_trace(go.Box(y=u_e_array_single_wrong, name='Wrong predictions'))
        fig.update_layout(title_text="Model {} - {} - {} - Epistemic uncertainty".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Box(y=u_t_array_single, name='Total uncertainty'))
        fig.update_layout(title_text="Model {} - {} - {}".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Box(y=u_a_array_single, name='Aleatoric uncertainty'))
        fig.update_layout(title_text="Model {} - {} - {}".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')

        fig = go.Figure()
        fig.add_trace(go.Box(y=u_e_array_single, name='Epistemic uncertainty'))
        fig.update_layout(title_text="Model {} - {} - {}".format(
            model_number, model_type.capitalize(), ds_name.capitalize()))
        fig.show(renderer='chromium')
