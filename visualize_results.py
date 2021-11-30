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
from utils import combine_two_string, reliability_diagram, idx_to_int, accuracy_function
from utils import single_accuracies, get_targets_probability, compute_features
from utils import compute_bin_data, import_dataset, load_models, process_args, compute_distributions
from utils import binary_crossentropy, max_multiclass_crossentropy, expected_calibration_error 
from plot_utils import accuracy_uncertainty_plot, proportions_plot
from plot_utils import mean_accuracy_plot, distributions_plot, box_plot_func
from plot_utils import sequences_plot, reliability_diagram_plot, distributions_plot_right_wrong
from plot_utils import event_correctness_plot, event_probability_plot
#from sklearn.stats import linregress
from statsmodels.stats.weightstats import DescrStatsW

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='choose the dataset',
                    choices=['helpdesk', 'bpic2012'])
parser.add_argument('model_directory', help='directory where the ensamble models are saved')
parser.add_argument('--model_calibrated', 
                    help='load model that have been calibrated. Default to False', 
                    default=False, type=bool)
parser.add_argument('--plot_entire_sequences',
                    help='plot the output distribution of N random sequences. Default to 0',
                    default=0, type=int)
parser.add_argument('--plot_wrong_predictions',
                    help='plot N output distribution of wrong predictions. Default to 0',
                    default=0, type=int)
parser.add_argument('--dataset_type',
                    help='choose what segment of dataset to use between training, validation and test. Default to all',
                    default='all', choices=['training', 'validation', 'test', 'all'])
parser.add_argument('--model_type',
                    help='choose the type of algorithm used to estimate uncertainty: MC dropout or ensamble. Default to ensamble',
                    default='ensamble', choices=['MC-dropout', 'ensamble'])
parser.add_argument('--samples_number',
                    help='number of sample for the MC dropout. Default to 5',
                    default=5, type=int)
parser.add_argument('--batch_size',
                    help='size of batches. Default to 64',
                    default=64, type=int)
parser.add_argument('--uncertainty_threshold',
                    help='uncertainty threshold to select cases. Default to 0.4', 
                    default=0.4, type=float)
parser.add_argument('--plot_cases_threshold',
                    help='plot cases below the threshold. Default to False', 
                    default=False, type=bool)
parser.add_argument('--save_cases_threshold',
                    help='save cases below the threshold. Default to False', 
                    default=False, type=bool)
parser.add_argument('--plot_mean_accuracy_vs_uncertainty',
                    help='plot mean seqeunces accuracy vs uncertainty. Default to False', 
                    default=False, type=bool)
parser.add_argument('--plot_event_probability_vs_uncertainty',
                    help='plot single event accuracy vs uncertainty. Default to True', 
                    default=True, type=bool)
parser.add_argument('--plot_event_correctness_vs_uncertainty',
                    help='plot single event correctness vs uncertainty. Default to False', 
                    default=False, type=bool)
parser.add_argument('--plot_box_plot_uncertainty',
                    help='plot box plot of uncertainties. Default to False', 
                    default=False, type=bool)
parser.add_argument('--plot_distributions',
                    help='plot distribution of uncertainties. Default to False', 
                    default=False, type=bool)
parser.add_argument('--plot_reliability_diagram',
                    help='plot reliability diagram. Default to True', 
                    default=True, type=bool)
parser.add_argument('--plot_accuracy_vs_uncertainty',
                    help='box plot of accuracy vs uncertainty. Default to True', 
                    default=True, type=bool)
parser.add_argument('--plot_proportions', 
                    help='box plot of right and wrong predictions normalize with the percentage of data in bin. Default to False', 
                    default=False, type=bool)
parser.add_argument('--tfds_id',
                    help='extract also the case id from dataset. Default to True', 
                    default=True, type=bool)

# Parse and check arguments
args = parser.parse_args()
plot_entire_seqs, plot_wrong_preds, dropout = process_args(parser)
dataset = args.dataset
model_dir = os.path.join('models_ensamble', dataset, args.model_directory)
model_number = args.model_directory.split('_')[1]

num_seq_entire = args.plot_entire_sequences
num_wrong_preds = args.plot_wrong_predictions
ds_type = args.dataset_type
model_type = args.model_type
num_samples = args.samples_number
# thresholds
unc_threshold = args.uncertainty_threshold
plot_threshold = args.plot_cases_threshold
save_threshold = args.save_cases_threshold
batch_size = args.batch_size
acc_threshold = 0.5
# plot stats
model_calibrated = args.model_calibrated
model_calibrated_title = 'Not Calibrated'
if model_calibrated:
    model_dir = os.path.join(model_dir, 'calibrated')
    model_calibrated_title = 'Calibrated'

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
datasets, builder_ds = import_dataset(dataset, ds_type, tfds_id, batch_size)
features, output_preprocess, models, vocabulary_act = load_models(model_dir, dataset, tfds_id, model_type)

count_seq = 0
count_wrong = 0
for ds, num_examples, ds_name in datasets:
    title_text='Model {} - {} - {} - {} - {}'.format(
        model_number, model_type.capitalize(), ds_name.capitalize(), dataset.capitalize(), model_calibrated_title)
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

        u_a, out_prob_tot_distr, out_prob = compute_distributions(
            models, model_type, input_data, dropout, num_samples)

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
                with open(os.path.join(model_dir, 'saved_cases_threshold_{}.txt'.format(np.round(unc_threshold, 4))), 'a') as file:
                    file.write("{}\n".format(case_selected[num_row, 0]))
            #breakpoint()

        if check_cond_wrong or check_cond_entire or (check_cond_tot_unc_prob and (save_threshold or plot_threshold)):
            sequences_plot(prob_unc_mask, acc_single, check_cond_wrong, random_idx_plot, input_data, u_t,
                           u_a, batch_size, plot_threshold, target_data, mask, plot_wrong_preds,
                           vocabulary_plot, out_prob, out_prob_tot_distr, model_type, 
                           batch_idx, title_text, count_wrong, count_seq, num_wrong_preds)

    if plot_reliability_diagram:
        ece_ensemble = expected_calibration_error(rel_dict)
        ece_one_model = expected_calibration_error(rel_dict_one_model)
        reliability_diagram_plot(rel_dict, rel_dict_one_model, rel_bins, title_text, ece_ensemble, ece_one_model)

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

    u_t_array_single = u_t_array_single[acc_array_single>-1]
    u_a_array_single = u_a_array_single[acc_array_single>-1]
    u_e_array_single = u_e_array_single[acc_array_single>-1]
    target_prob_array = target_prob_array[acc_array_single>-1]
    target_label_array = target_label_array[acc_array_single>-1]
    target_case_array = target_case_array[acc_array_single>-1]
    max_prob_event_array = max_prob_event_array[acc_array_single>-1]
    acc_array_single = acc_array_single[acc_array_single>-1]
    if plot_distr:
        distributions_plot(u_t_array_single, u_a_array_single, u_e_array_single, title_text)

    if plot_mean_acc:
        mean_accuracy_plot(u_t_array, acc_array, 'Total', title_text)
        mean_accuracy_plot(u_a_array, acc_array, 'Aleatoric', title_text)
        mean_accuracy_plot(u_e_array, acc_array, 'Epistemic', title_text)

# select valid unceratinties and divide between right and wrong predictions
    u_t_array_single_right = u_t_array_single[acc_array_single == 1]
    u_t_array_single_wrong = u_t_array_single[acc_array_single == 0]
    u_a_array_single_right = u_a_array_single[acc_array_single == 1]
    u_a_array_single_wrong = u_a_array_single[acc_array_single == 0]
    u_e_array_single_right = u_e_array_single[acc_array_single == 1]
    u_e_array_single_wrong = u_e_array_single[acc_array_single == 0]
    # Compute bins to highligth percentage of wrong and right prediction
    u_t_plot, perc_right_plot, perc_wrong_plot, perc_data, acc_plot = compute_bin_data(
        u_t_array_single, acc_array_single)

    if plot_proportions:
        proportions_plot(u_t_plot, perc_rigth_plot, perc_wrong_plot, title_text)

    if plot_acc_unc:
        data = np.asarray([np.cumsum(u_t_plot)-u_t_plot, acc_plot]).T
        results = DescrStatsW(data, perc_data)
        corr_coeff = results.corrcoef
        accuracy_uncertainty_plot(u_t_plot, acc_plot, perc_data, '{} - Corr Coeff {}'.format(title_text, np.round(corr_coeff[0,1], 4)))

    prob_array = np.linspace(1e-6, 1-(1e-6), 500)
    bin_entropy_array = binary_crossentropy(prob_array)
    multi_entropy_arrays = []
    for i in range(len(vocabulary_act), 1, -1):
        multi_entropy_arrays.append(max_multiclass_crossentropy(prob_array,
                                                                num_class=i))
    prob = np.linspace(1/2, 1/3, 100)
    prob2 = np.linspace(1-(1e-6), 0.5, 100)
    #breakpoint()
    entropy_other = prob*np.log(prob) + prob2*(1 - prob)*np.log(prob2*(1 - prob)) + \
        (1 - prob2)*(1 - prob) * np.log((1 - prob2)*(1 - prob))
    entropy_other = -entropy_other
    if plot_event_prob:
        total_label = list(map(combine_two_string, target_case_array, target_label_array))
        event_probability_plot(total_label, u_t_array_single_right, u_t_array_single_wrong,
                               target_prob_array, acc_array_single, multi_entropy_arrays, prob_array,
                               vocabulary_act, bin_entropy_array, u_a_array_single_right,
                               u_a_array_single_wrong, u_e_array_single_right, u_e_array_single_wrong,
                               max_prob_event_array, title_text, entropy_other, target_label_array, prob)
    if plot_event_corr:
        event_correctness_plot(u_t_array_single, u_t_array_single, u_e_array_single,
                               acc_array_single, target_label_array, target_prob_array, title_text)
# Plot distribution
    if plot_distr:
        distributions_plot_right_wrong(u_t_array_single_right, u_t_array_single_wrong,
                                       u_a_array_single_right, u_a_array_single_wrong,
                                       u_e_array_single_right, u_e_array_single_wrong, title_text)
# Boxplot
    if plot_box_plot:
        box_plot_func(u_t_array_single_right, u_t_array_single_wrong, u_t_array_single,
                      u_a_array_single_right, u_t_array_single_wrong, u_a_array_single,
                      u_e_array_single_right, u_e_array_single_wrong, u_e_array_single,
                      title_text)
