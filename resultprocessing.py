import tensorflow as tf
import numpy as np
import pm4py
from pathlib import Path
from utilities.generals import accuracy_function, reliability_diagram 
from utilities.generals import single_accuracies, accuracy_top_k, get_targets_probability
from utilities.generals import get_variants_percentage, get_case_percentage, extract_case, filter_variant
from utilities.generals import get_variant_characteristics, get_case_statistics, get_case_characteristic

K_TOP = 6


class ResultProcessor:
    def __init__(self, unc_threshold: float=0.4, chosen_case: str=""):
        self.accuracy_function = accuracy_function
        self.acc_threshold = 0.5
        self.unc_threshold = unc_threshold
        self.rel_dict = {}
        self.rel_dict_one_model = {}
        self.max_prob_event_array = np.zeros(1)
        self.target_prob_array = np.zeros(1)
        self.u_t_array_single = np.zeros(1)
        self.acc_array_single = np.zeros(1)
        self.target_label = np.asarray(['0'])
        self.target_case = np.asarray(['0'])
        self.case_selected_numpy = []
        self.acc_single_model = []
    #acc_top_k_array = list()
        self.acc_top_k_dict = {k_class: [] for k_class in range(1, K_TOP+1)}
        self.dict_predict_case = {}
        self.chosen_case = chosen_case

    def compute_reliability_diagram_one_model(self, target_data, out_prob, rel_bins=0.05):
        self.rel_dict_one_model = reliability_diagram(
                target_data.numpy(), out_prob.numpy(), self.rel_dict_one_model, rel_bins)

    def compute_reliability_diagram(self, target_data, out_prob, rel_bins=0.05):
        self.rel_dict = reliability_diagram(
                target_data.numpy(), out_prob.numpy(), self.rel_dict, rel_bins)

    def get_case_below_threshold(self, target_data_case, acc_single, u_t, mask):
        prob_unc_mask = np.logical_and(tf.math.less(
            acc_single, self.acc_threshold).numpy(), u_t<self.unc_threshold)
        prob_unc_mask = np.logical_and(prob_unc_mask, mask)
        prob_unc_mask = np.ma.masked_equal(prob_unc_mask, True)
        prob_unc_mask = np.ma.mask_rows(prob_unc_mask).mask

        #breakpoint()
        case_selected = target_data_case[prob_unc_mask]
        #for num_row in range(case_selected.shape[0]):
        if case_selected.size != 0:
            case_selected = case_selected.reshape((-1, prob_unc_mask.shape[1]))
            self.case_selected_numpy = np.hstack((self.case_selected_numpy,
                                                    np.asarray(case_selected)[:, 0]))

    def overconfidence_analysis(self, case_id_train, event_log, features, dataset_name):
        features_type = features['features-type']
        features_variant = features['features-variant']
        #event_log_train = pm4py.filter_log(lambda x: x[0]['Case ID'].split()[1] in case_id_train, event_log) 
        event_log_train = pm4py.filter_log(lambda x: x.attributes['concept:name'] in case_id_train, event_log) 
        total_variants_perc, total_variants = get_variants_percentage(event_log_train)
        variant_case_dict = dict()
        total_selected_variants = set()
        for case in self.case_selected_numpy:
            variant, percentage = get_case_percentage(event_log, case, total_variants_perc)
            if not variant in variant_case_dict:
                variant_case_dict[variant] = []
            total_selected_variants.add((variant, percentage))
            variant_case_dict[variant].append(case)
        
        total_variants_stats = dict()
        count_variant_in_training = 0
        count_perc_variant = 0
        count_perc_variant2 = 0
        for variant in total_selected_variants:
            if variant[1] > 0:
                if variant[1] > 0.001:
                    count_perc_variant += 1
                if variant[1] > 0.005:
                    count_perc_variant2 += 1
                count_variant_in_training += 1
                total_variants_stats[variant[0]] = dict()
                for feature in features_variant:
                    if features_type[feature] == 'categorical':
                        total_variants_stats[variant[0]][feature] = list()
                    elif features_type[feature] == 'continuous':
                        total_variants_stats[variant[0]][feature] = list()
#                total_variants_stats[variant[0]]['day_part'] = dict()
#                total_variants_stats[variant[0]]['week_day'] = dict()
                total_variants_stats[variant[0]]['completion_time'] = list()
                filtered_log = filter_variant(event_log, variant[0])
                features_dict = get_variant_characteristics(filtered_log, features_variant, features_type)
                #breakpoint()
                for case_id in variant_case_dict[variant[0]]:
                    case_seq = extract_case(case_id, event_log, dataset_name)
                    case_dict = get_case_characteristic(case_seq, features_variant, features_type)
                    #breakpoint()
                    case_stats = get_case_statistics(case_dict, features_dict, features_type, case_seq)
                    #breakpoint()
                    for feature in total_variants_stats[variant[0]]:
                        if isinstance(case_stats[feature], dict):
                            for value in case_stats[features_variant]:
                                if isinstance(total_variants_stats[variant[0]][feature], dict):
                                    if not value in total_variants_stats[variant[0]][feature]:
                                        total_variants_stats[variant[0]][feature][value] = list(
                                                [case_stats[feature][value]])
                                    else:
                                        total_variants_stats[variant[0]][feature][value].append(
                                                case_stats[feature][value])
                        else:
                            total_variants_stats[variant[0]][feature].append(case_stats[feature])
        #breakpoint()
        variant_perc_dict = {elem[0]: elem[1] for elem in total_selected_variants}
        features_stats = dict()
        features_stats['perc'] = list()
        var_thres_perc = 0.005
        for variant in total_variants_stats:
            if variant_perc_dict[variant] > var_thres_perc:
                for feature in total_variants_stats[variant]:
                    if not feature in features_stats:
                        features_stats[feature] = list()
                    if isinstance(total_variants_stats[variant][feature], dict):
                        for value in total_variants_stats[variant][feature]:
                            features_stats[feature].extend(total_variants_stats[variant][feature][value])
                    else:
                        features_stats[feature].extend(total_variants_stats[variant][feature])
                features_stats['perc'].append(variant_perc_dict[variant])

        perc_features = {}
        quartile_features = {}
        for feature in features_stats:
            if not feature == 'perc':
                if feature == 'completion_time' or features_type[feature] == 'continuous' :
                    perc_quart = np.sum(np.less(features_stats[feature], 25))
                    perc_quart += np.sum(np.greater(features_stats[feature], 75))
                    perc_quart /= len(features_stats[feature])
                    quartile_features[feature] = perc_quart
                else:
                    perc_features[feature] = np.mean(features_stats[feature])
        self.overconfidence_results = {'perc_features': perc_features, 
                                       'quartile_features': quartile_features,
                                       'perc_test_in_train': count_variant_in_training/len(total_selected_variants),
                                       'perc_test_gt_001': count_perc_variant/count_variant_in_training,
                                       'perc_test_gt_005': count_perc_variant2/count_variant_in_training}
        #breakpoint()



    def process_batch(self, target_data, target_data_case, out_prob, out_prob_tot_distr, u_t, input_data):
        self.target_label = np.hstack([self.target_label, target_data.numpy().ravel()])
        self.target_case = np.hstack([self.target_case, target_data_case.ravel()])

        # reliability diagrams
        self.compute_reliability_diagram_one_model(target_data, out_prob)
        self.compute_reliability_diagram(target_data, out_prob_tot_distr)

        # compute accuracies
        acc_single = single_accuracies(target_data, out_prob_tot_distr)
        self.acc_array_single = np.hstack([self.acc_array_single, acc_single.numpy().ravel()])
        self.acc_single_model.append(accuracy_top_k(target_data, out_prob, k=1))
        for k_class in self.acc_top_k_dict.keys():
            acc_top_k = accuracy_top_k(target_data, out_prob_tot_distr, k=k_class)
            self.acc_top_k_dict[k_class].append(acc_top_k)

        # compute probabilities
        max_prob_event = tf.reduce_max(out_prob_tot_distr, axis=2)
        self.max_prob_event_array = np.hstack([self.max_prob_event_array, max_prob_event.numpy().ravel()])
        
        target_prob = get_targets_probability(target_data, out_prob_tot_distr)
        self.target_prob_array = np.hstack(
                [self.target_prob_array, target_prob.numpy().ravel()])

        # compute uncertainty
        mask = tf.math.logical_not(tf.math.equal(target_data, 0))
        u_t = np.where(mask, u_t, np.zeros(np.shape(u_t)))
        self.u_t_array_single = np.hstack([self.u_t_array_single, u_t.ravel()])
        self.get_case_below_threshold(target_data_case, acc_single, u_t, mask)
        
        # save elements to plot predicted cases if present
        if self.chosen_case in self.target_case:
            self.dict_predict_case = {'acc_single': acc_single, 'input_data': input_data, 'u_t': u_t,
                                      'target_data': target_data, 'out_prob': out_prob, 'out_prob_tot_distr': out_prob_tot_distr,
                                      'target_data_case': target_data_case, 'chosen_case': self.chosen_case}
#            plot_case(acc_single, input_data, u_t, target_data, vocabulary_plot,
#                    out_prob, out_prob_tot_distr, model_type, title_text, target_data_case, predict_case)

    def process_after_batches(self):
        self.acc_array_single = self.acc_array_single[1:]
        self.u_t_array_single = self.u_t_array_single[1:]
        self.target_prob_array = self.target_prob_array[1:]
        self.target_label = self.target_label[1:]
        self.target_case = self.target_case[1:]
        self.max_prob_event_array = self.max_prob_event_array[1:]
        #breakpoint()
        # ordering
        ordered_acc_array = self.acc_array_single.argsort()
        self.acc_array_single = self.acc_array_single[ordered_acc_array]
        self.u_t_array_single = self.u_t_array_single[ordered_acc_array]
        self.target_prob_array = self.target_prob_array[ordered_acc_array]
        self.target_label_array = self.target_label[ordered_acc_array]
        self.max_prob_event_array = self.max_prob_event_array[ordered_acc_array]
        self.target_case_array = self.target_case[ordered_acc_array]
        #breakpoint()

        # select valid predictions
        self.u_t_array_single = self.u_t_array_single[self.acc_array_single>-1]
        self.target_prob_array = self.target_prob_array[self.acc_array_single>-1]
        self.target_label_array = self.target_label_array[self.acc_array_single>-1]
        self.target_case_array = self.target_case_array[self.acc_array_single>-1]
        self.max_prob_event_array = self.max_prob_event_array[self.acc_array_single>-1]
        self.acc_array_single = self.acc_array_single[self.acc_array_single>-1]

    def get_uncertainty_results(self):
        # select valid unceratinties and divide between right and wrong predictions
        u_t_array_single_right = self.u_t_array_single[self.acc_array_single == 1]
        u_t_array_single_wrong = self.u_t_array_single[self.acc_array_single == 0]
        return {'u_t_array_single': self.u_t_array_single, 
                'u_t_array_single_right': u_t_array_single_right,
                'u_t_array_single_wrong': u_t_array_single_wrong}

    def get_accuracy_results(self):
        return {'acc_array_single': self.acc_array_single, 'top_k': self.acc_top_k_dict}

    def get_target_results(self):
        return {'target_label_array': self.target_label_array, 'target_case_array': self.target_case_array}

    def get_probabilities_results(self):
        return {'target_prob_array': self.target_prob_array, 'max_prob_event_array': self.max_prob_event_array}

    def get_predict_case_results(self):
        return self.dict_predict_case

    def get_reliability_diagram_results(self):
        return {'rel_dict': self.rel_dict, 'rel_dict_one_model': self.rel_dict_one_model}

    def get_overconfidence_analysis(self):
        return self.overconfidence_results

    def write_top_k_accuracy_results(self, model_dir, model_name):
        with open(model_dir / "{}-top_{}_accuracies.txt".format(model_name, K_TOP), "a") as file:
            for k_class in self.acc_top_k_dict.keys():
                print("Top {} accuracy: {}".format(k_class, np.asarray(self.acc_top_k_dict[k_class]).mean()))
                file.write("Top {} accuracy: {}\n".format(k_class, np.asarray(self.acc_top_k_dict[k_class]).mean()))

    def write_overconfidence_analysis_results(self, model_dir: Path):
        with open(model_dir / 'overconfidence.results', 'w') as file:
            file.write("Results of the analysis of overconfident wrong predictions\n\n")
            file.write("Percentage of test variants also present in training set: ")
            file.write(f"{np.round(self.overconfidence_results['perc_test_in_train'], 5)}\n")
            file.write("Percentage of test variants present in training set more than 0.1%: ")
            file.write(f"{np.round(self.overconfidence_results['perc_test_gt_001'], 5)}\n")
            file.write("Percentage of test variants present in training set more than 0.5%: ")
            file.write(f"{np.round(self.overconfidence_results['perc_test_gt_005'], 5)}\n\n")
            file.write("Considering test variants in training test wwith more than 0.5%:\n")
            for feature in self.overconfidence_results['quartile_features']:
                file.write(f"Percentage of cases with values of feature '{feature}' in first/last quartile: ")
                file.write(f"{np.round(self.overconfidence_results['quartile_features'][feature], 5)}\n") 
            for feature in self.overconfidence_results['perc_features']:
                file.write(f"Mean percentage of values of feature '{feature}' of the case ")
                file.write("with respect to the overall cases belonging to the same variant: ") 
                file.write(f"{np.round(self.overconfidence_results['perc_features'][feature], 5)}\n")

    def write_case_selected_numpy(self, model_dir, set_name):
        with open(model_dir / f'saved_cases_threshold_{np.round(self.unc_threshold, 4)}_{set_name}.npy', 'wb') as file:
            np.save(file, self.case_selected_numpy)
