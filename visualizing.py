import os
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from utilities.generals import expected_calibration_error, compute_bin_data
from utilities.generals import max_multiclass_crossentropy, combine_two_string
from utilities.plots import reliability_diagram_plot, accuracy_uncertainty_plot, event_probability_plot, plot_case


class ResultsPlotter:
    def __init__(self, results, title_text, figures_dir):
        self.uncertainty = results['uncertainty']
        self.probability = results['probability']
        self.target = results['target']
        self.case = results['case']
        self.reliability = results['reliability']
        self.accuracy = results['accuracy']
        self.title_text = title_text
        self.figures_dir = figures_dir

    def reliability_diagram(self, rel_bins=0.05):
        ece_ensemble = expected_calibration_error(self.reliability['rel_dict'])
        ece_one_model = expected_calibration_error(self.reliability['rel_dict_one_model'])
        fig = reliability_diagram_plot(self.reliability['rel_dict'], self.reliability['rel_dict_one_model'],
                                       rel_bins, self.title_text, ece_ensemble, ece_one_model)
        self.save_fig(fig, 'reliability-diagram')

    def accuracy_uncertainty(self):
        u_t_plot, _, _, perc_data, acc_plot = compute_bin_data(
                self.uncertainty['u_t_array_single'], self.accuracy['acc_array_single'])
        data = np.asarray([np.cumsum(u_t_plot)-u_t_plot, acc_plot]).T
        results = DescrStatsW(data, perc_data)
        corr_coeff = results.corrcoef
        fig = accuracy_uncertainty_plot(u_t_plot, acc_plot, perc_data, 
                                        '{} - Corr Coeff {}'.format(self.title_text, np.round(corr_coeff[0,1], 4)))
        self.save_fig(fig, 'accuracy-uncertainty')

    def event_probability(self, vocabulary_act):
        prob_array = np.linspace(1e-6, 1-(1e-6), 500)
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
        total_label = list(map(
            combine_two_string, self.target['target_case_array'], self.target['target_label_array']))
        fig = event_probability_plot(total_label, self.uncertainty['u_t_array_single_right'],
                                     self.uncertainty['u_t_array_single_wrong'], self.probability['target_prob_array'],
                                     self.accuracy['acc_array_single'], multi_entropy_arrays, prob_array,
                                     vocabulary_act, self.title_text)
        self.save_fig(fig, 'event-probability-total')

    def case_prediction(self, vocabulary_act, model_type):
        vocabulary_plot = ['<PAD>']
        vocabulary_plot.extend(vocabulary_act)
        plot_case(self.case['acc_single'], self.case['input_data'], self.case['u_t'],
                  self.case['target_data'], vocabulary_plot, self.case['out_prob'],
                  self.case['out_prob_tot_distr'], model_type, self.title_text, 
                  self.case['target_data_case'], self.case['chosen_case'], self.figures_dir)

    def save_fig(self, fig, fig_name):
        if not self.figures_dir.exists():
            self.figures_dir.mkdir()
        fig.show(renderer='chromium')
        fig_file = self.figures_dir / f"{fig_name}-{''.join(self.title_text.split())}.svg"
        fig.write_image(fig_file)
        fig_file = self.figures_dir / f"{fig_name}-{''.join(self.title_text.split())}.html"
        fig.write_html(fig_file)
