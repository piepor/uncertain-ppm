import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import yaml
import tensorflow_datasets as tfds
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import argparse
import re
from utils import compute_features, loss_function, accuracy_function, compute_input_signature
from helpdesk_utils import helpdesk_utils
from bpic2012_utils import bpic2012_utils
from utils import combine_two_string, reliability_diagram, idx_to_int, accuracy_function
from utils import single_accuracies, get_targets_probability, compute_features
from utils import compute_bin_data, import_dataset, load_models
from utils import process_args, compute_distributions, anomaly_detection_isolation_forest
from utils import binary_crossentropy, max_multiclass_crossentropy, expected_calibration_error 
from utils import get_case_percentage, get_variants_percentage, filter_variant
from utils import get_variant_characteristics, get_case_statistics, extract_case
from utils import get_case_characteristic, accuracy_top_k
from utils_plot import accuracy_uncertainty_plot, proportions_plot
from utils_plot import mean_accuracy_plot, distributions_plot, box_plot_func
from utils_plot import sequences_plot, reliability_diagram_plot, distributions_plot_right_wrong
from utils_plot import event_correctness_plot, event_probability_plot, plot_case
#from sklearn.stats import linregress
from statsmodels.stats.weightstats import DescrStatsW
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
import pandas as pd
tf.keras.backend.clear_session()
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
parser.add_argument('--top_k_accuracy',
                    help='select k elements from output distribution for accuracy. Default to 6', 
                    default=6, type=int)
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
parser.add_argument('--anomaly_detection',
                    help='Run anomaly detection with Isolation Forest. Default to None', 
                    default=False, type=bool)
parser.add_argument('--variant_threshold',
                    help='Threshold on variant frequency in the event log for visualizing characteristics of anomalies. Default to 0.1', 
                    default=0.1, type=float)
parser.add_argument('--predict_case',
                    help='Case to predict', 
                    default=0, type=int)

# Parse and check arguments
args = parser.parse_args()
plot_entire_seqs, plot_wrong_preds, dropout = process_args(parser)
dataset = args.dataset
model_dir = os.path.join('models', dataset, args.model_directory)
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
variant_threshold = args.variant_threshold
acc_threshold = 0.5
k_top = args.top_k_accuracy
# plot stats
model_calibrated = args.model_calibrated
model_calibrated_title = 'Not Calibrated'

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
anomaly_detection = args.anomaly_detection
predict_case = args.predict_case

# Start analysis - import dataset
# change read config in order to return also the case id
datasets, builder_ds, case_id_train  = import_dataset(dataset, ds_type, tfds_id, batch_size)

features, output_preprocess, models, vocabulary_act, features_type, features_variant = load_models(model_dir, dataset, tfds_id, model_type)

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
    acc_array_mean = np.zeros(1)
    acc_array_single = np.zeros(1)
    target_prob_array = np.zeros(1)
    max_prob_event_array = np.zeros(1)
    target_label = np.asarray(['0'])
    target_case = np.asarray(['0'])
    rel_dict = dict()
    rel_dict_one_model = dict()
    case_selected_numpy = list()
    #acc_top_k_array = list()
    acc_top_k_dict = {k_class: list() for k_class in range(1, k_top+1)}
    acc_single_model = list()
    for batch_idx, batch_data in enumerate(ds):

        #breakpoint()
        input_data = []
        for feature in features:
            input_data.append(batch_data[feature['name']][:, :-1])
        target_data = batch_data['activity'][:, 1:]
        #breakpoint()
        #exs = [ex.numpy().decode('utf-8') for ex in batch_data['tfds_id']]
        exs = [ex.numpy()[0].decode('utf-8') for ex in batch_data['case_id']]
        #target_data_case = [idx_to_int(tfds_id, builder_ds) for tfds_id in exs] 
        target_data_case = [int(case) for case in exs] 
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
        acc_single_model.append(accuracy_top_k(target_data, out_prob, k=1))
        for k_class in acc_top_k_dict.keys():
            acc_top_k = accuracy_top_k(target_data, out_prob_tot_distr, k=k_class)
            acc_top_k_dict[k_class].append(acc_top_k)        
        #breakpoint()
        target_prob = get_targets_probability(target_data, out_prob_tot_distr)
        target_prob_array = np.hstack([target_prob_array, target_prob.numpy().ravel()])

        # compute total uncertainty
        u_t = -np.sum(out_prob_tot_distr.numpy() * np.log(out_prob_tot_distr.numpy()), axis=-1)
        u_t = np.where(mask, u_t, np.zeros(np.shape(u_t)))
        u_t_array_single = np.hstack([u_t_array_single, u_t.ravel()])

        vocabulary_plot = ['<PAD>']
        vocabulary_plot.extend(vocabulary_act)

        length_seq = tf.reduce_sum(
                tf.cast(tf.math.logical_not(
                    tf.math.equal(target_data, 0)), dtype=tf.float32),
                axis=-1)
        mean_u_t = np.sum(u_t, axis=-1) / length_seq.numpy()

        u_t_array_mean = np.hstack([u_t_array_mean, mean_u_t])

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
            #for num_row in range(case_selected.shape[0]):
            if case_selected.size != 0:
                case_selected = case_selected.reshape((-1, prob_unc_mask.shape[1]))
                case_selected_numpy = np.hstack((case_selected_numpy, 
                                                 np.asarray(case_selected)[:, 0]))
        #if check_cond_wrong or check_cond_entire or (check_cond_tot_unc_prob and (save_threshold or plot_threshold)):
        if predict_case in target_case[1:]:
            plot_case(acc_single, input_data, u_t, target_data, vocabulary_plot,
                    out_prob, out_prob_tot_distr, model_type, title_text, target_data_case, predict_case)

    with open(os.path.join(
        model_dir, 'saved_cases_threshold_{}_{}.npy'.format(
            np.round(unc_threshold, 4), ds_name.split()[0])), 'wb') as file:
        np.save(file, case_selected_numpy)

    if plot_reliability_diagram:
        ece_ensemble = expected_calibration_error(rel_dict)
        ece_one_model = expected_calibration_error(rel_dict_one_model)
        reliability_diagram_plot(rel_dict, rel_dict_one_model, rel_bins, title_text, ece_ensemble, ece_one_model)

    u_t_array = u_t_array_mean[1:]
    acc_array = acc_array_mean[1:]
    acc_array_single = acc_array_single[1:]
    u_t_array_single = u_t_array_single[1:]
    target_prob_array = target_prob_array[1:]
    target_label = target_label[1:]
    target_case = target_case[1:]
    max_prob_event_array = max_prob_event_array[1:]
    #breakpoint()
# ordering
    ordered_acc_array = acc_array_single.argsort()
    acc_array_single = acc_array_single[ordered_acc_array]
    u_t_array_single = u_t_array_single[ordered_acc_array]
    target_prob_array = target_prob_array[ordered_acc_array]
    target_label_array = target_label[ordered_acc_array]
    max_prob_event_array = max_prob_event_array[ordered_acc_array]
    target_case_array = target_case[ordered_acc_array]
    #breakpoint()

    u_t_array_single = u_t_array_single[acc_array_single>-1]
    target_prob_array = target_prob_array[acc_array_single>-1]
    target_label_array = target_label_array[acc_array_single>-1]
    target_case_array = target_case_array[acc_array_single>-1]
    max_prob_event_array = max_prob_event_array[acc_array_single>-1]
    acc_array_single = acc_array_single[acc_array_single>-1]

# select valid unceratinties and divide between right and wrong predictions
    u_t_array_single_right = u_t_array_single[acc_array_single == 1]
    u_t_array_single_wrong = u_t_array_single[acc_array_single == 0]
    # Compute bins to highligth percentage of wrong and right prediction
    u_t_plot, perc_right_plot, perc_wrong_plot, perc_data, acc_plot = compute_bin_data(
        u_t_array_single, acc_array_single)

    if plot_acc_unc:
        data = np.asarray([np.cumsum(u_t_plot)-u_t_plot, acc_plot]).T
        results = DescrStatsW(data, perc_data)
        corr_coeff = results.corrcoef
        #breakpoint()
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

    if anomaly_detection:
        #breakpoint()
        mask_case = np.in1d(anomalies, target_case)
        masked_anomalies = np.ma.masked_array(anomalies, mask=~mask_case)
        ds_anomalies = masked_anomalies.compressed()

        with open(os.path.join(
            model_dir, 'anomalies_IF_{}.npy'.format(ds_name.split()[0])), 'wb') as file:
            np.save(file, ds_anomalies)
        common_anomalies_mask = np.in1d(case_selected_numpy, ds_anomalies)
        masked_common_anomalies = np.ma.masked_array(case_selected_numpy, 
                                                     mask=~common_anomalies_mask)
        common_anomalies = masked_common_anomalies.compressed()
        common_variants = set()
        total_selected_variants = set()
        anomalous_variants = set()
        total_variants_perc, total_variants = get_variants_percentage(event_log_train)
        variant_case_dict = dict()
        for case in case_selected_numpy:
            #breakpoint()
            variant, percentage = get_case_percentage(event_log, case, total_variants_perc)
            variant_case_dict[variant] = []

        for common_anomaly in common_anomalies:
            variant, percentage = get_case_percentage(event_log, common_anomaly, total_variants_perc)
            common_variants.add((variant, percentage))
        for case in case_selected_numpy:
            #variant, percentage = get_case_percentage(event_log, case, total_variants_perc)
            variant, percentage = get_case_percentage(event_log, case, total_variants_perc)
            total_selected_variants.add((variant, percentage))
            variant_case_dict[variant].append(case)
        other_selected_variants = total_selected_variants.difference(common_variants)
        total_variants_stats = dict()
        #breakpoint()
        count_variant_in_training = 0
        count_perc_variant = 0
        count_perc_variant2 = 0
        for variant in total_selected_variants:
            #if variant[1] > variant_threshold or variant[1] == 0:
            #breakpoint()
            if variant[1] > 0:
                if variant[1] > 0.001:
                    count_perc_variant += 1
                if variant[1] > 0.005:
                    count_perc_variant2 += 1
                count_variant_in_training += 1
                total_variants_stats[variant[0]] = dict()
                for feature in features_variant.keys():
                    if features_type[feature] == 'categorical':
                        total_variants_stats[variant[0]][feature] = dict()
                    elif features_type[feature] == 'continuous':
                        total_variants_stats[variant[0]][feature] = list()
                total_variants_stats[variant[0]]['day_part'] = dict()
                total_variants_stats[variant[0]]['week_day'] = dict()
                total_variants_stats[variant[0]]['completion_time'] = list()
                #breakpoint()
                filtered_log = filter_variant(event_log, variant[0])
#                features_type = {'Resource': 'categorical', 'product': 'categorical',
#                                 'seriousness_2': 'categorical', 'responsible_section': 'categorical',
#                                 'service_level': 'categorical', 'service_type': 'categorical',
#                                 'workgroup': 'categorical', 'day_part': 'categorical', 'week_day': 'categorical'}
#                features_variant = {'Resource':'event', 'product':'event', 'seriousness_2':'event', 'responsible_section': 'event',
#                        'service_level': 'event', 'service_type': 'event', 'workgroup':'event'}
                features_dict = get_variant_characteristics(filtered_log, features_variant, features_type)
                for case_id in variant_case_dict[variant[0]]:
                    #case_id = variant_case_dict[variant[0]][0]
                    case_seq = extract_case(case_id, event_log, dataset)
                    case_dict = get_case_characteristic(case_seq, features_variant, features_type)
                    case_stats = get_case_statistics(case_dict, features_dict, features_type, case_seq)
                    for feature in total_variants_stats[variant[0]].keys():
                        if isinstance(case_stats[feature], dict):
                            for value in case_stats[feature].keys():
                                if isinstance(total_variants_stats[variant[0]][feature], dict):
                                    if not value in total_variants_stats[variant[0]][feature].keys():
                                        total_variants_stats[variant[0]][feature][value] = list([case_stats[feature][value]])
                                    else:
                                        total_variants_stats[variant[0]][feature][value].append(case_stats[feature][value])
                        else:
                            total_variants_stats[variant[0]][feature].append(case_stats[feature])
        #breakpoint()
        variant_perc_dict = {elem[0]:elem[1] for elem in total_selected_variants}
        for variant in total_variants_stats.keys():
            if variant_perc_dict[variant] > variant_threshold:
                n_rows = int(np.ceil(len(total_variants_stats[variant].keys()) / 2))
                fig = make_subplots(rows=n_rows, cols=2, subplot_titles=list(total_variants_stats[variant].keys()))
                count = 0
                max_y = 0
                for feature in total_variants_stats[variant].keys():
                    row = int(np.floor(count/2)) + 1
                    col = count%2 + 1
                    if isinstance(total_variants_stats[variant][feature], dict):
                        for value in total_variants_stats[variant][feature].keys():
                            #breakpoint()
                            if max(total_variants_stats[variant][feature][value]) > max_y:
                                max_y = max(total_variants_stats[variant][feature][value]) + 0.2
                            fig.add_trace(go.Box(y=total_variants_stats[variant][feature][value], 
                                name=value), row=row, col=col)
                    else:
                        bin_size = 5
                        fig.add_trace(go.Histogram(x=total_variants_stats[variant][feature], xbins=dict(size=bin_size)), row=row, col=col)
                        #breakpoint()
                    count += 1
                count = 0
                for feature in total_variants_stats[variant].keys():
                    row = int(np.floor(count/2)) + 1
                    col = count%2 + 1
                    if not feature == 'completion_time':
                        if features_type[feature] == 'categorical':
                            fig.update_yaxes(range=[0, max_y], row=row, col=col)
                        elif features_type[feature] == 'continuous':
                            fig.update_xaxes(range=[0, 100+bin_size], row=row, col=col)
                    else:
                        fig.update_xaxes(range=[0, 100+bin_size], row=row, col=col)
                    count += 1
                title = 'Variant frequence in training event log: {} - anomalous cases in test set: {}'.format(np.round(variant_perc_dict[variant], 4), 
                        len(variant_case_dict[variant]))
                fig.update_layout(showlegend=False, title_text=title)
                fig.show(renderer='chromium')
            #breakpoint()
        # extract feature stats
        #breakpoint()
        features_stats = dict()
        features_stats['perc'] = list()
        var_thres_perc = 0.001
        for variant in total_variants_stats.keys():
            if variant_perc_dict[variant] > var_thres_perc:
                for feature in total_variants_stats[variant].keys():
                    if not feature in features_stats.keys():
                        features_stats[feature] = list()
                    if isinstance(total_variants_stats[variant][feature], dict):
                        for value in total_variants_stats[variant][feature].keys():
                            features_stats[feature].extend(total_variants_stats[variant][feature][value])
                    else:
                        features_stats[feature].extend(total_variants_stats[variant][feature])
                features_stats['perc'].append(variant_perc_dict[variant])
        #breakpoint()
        bin_size = 0.05
        for feature in features_stats.keys():
            #if feature == 'completion_time':
            fig = go.Figure() 
            if feature == 'perc':
                bin_size = 0.001
            elif feature == 'completion_time':
                bin_size = 5
                perc_quart = np.sum(np.less(features_stats[feature], 25))
                perc_quart += np.sum(np.greater(features_stats[feature], 75))
                perc_quart /= len(features_stats[feature])
                print('percentage time completion in first/last quartile: {}'.format(perc_quart))
            else:
                if features_type[feature] == 'continuous':
                    bin_size = 5
                    perc_quart = np.sum(np.less(features_stats[feature], 25))
                    perc_quart += np.sum(np.greater(features_stats[feature], 75))
                    perc_quart /= len(features_stats[feature])
                    print('percentage {} in first/last quartile: {}'.format(feature, perc_quart))
                elif features_type[feature] == 'categorical':
                    bin_size = 0.05
            fig.add_trace(go.Histogram(x=features_stats[feature], xbins=dict(size=bin_size)))
            #fig.add_trace(go.Histogram(x=features_stats[feature]))
            fig.update_layout(title_text=feature)
            fig.show(renderer='chromium')
            fig.write_html('{}-{}-{}.html'.format(feature, var_thres_perc, title_text))
        #breakpoint()
#        case_id = 4561
#        out_ensamble, out_single, target = predict_case(models, case_id, ds, builder_ds, dropout, num_samples, features, model_type)
#        print(len(common_anomalies)/len(ds_anomalies))
#        print(len(common_anomalies)/len(case_selected_numpy))
if anomaly_detection:
    print("percentage test variant in training: {}".format(count_variant_in_training/len(total_selected_variants)))
    print("percentage test variant in training more than 0.001: {}".format(count_perc_variant/count_variant_in_training))
    print("percentage test variant in training more than 0.005: {}".format(count_perc_variant/count_variant_in_training))

tf.keras.backend.clear_session()
with open(os.path.join(model_dir, "top_{}_accuracies.txt".format(k_top)), "a") as file:
    for k_class in acc_top_k_dict.keys():
        print("Top {} accuracy: {}".format(k_class, np.asarray(acc_top_k_dict[k_class]).mean()))
        file.write("Top {} accuracy: {}\n".format(k_class, np.asarray(acc_top_k_dict[k_class]).mean()))
print("SINGLE MODEL accuracy {}".format(np.asarray(acc_single_model).mean()))
