import numpy as np
from pathlib import Path
from tqdm import tqdm
from copy import copy
import tensorflow as tf
from tensorflow import keras as tfk
from dataprocessing import DataProcessorTF
from resultprocessing import ResultProcessor
from utilities.generals import compute_train_ids, import_event_log

K_TOP = 6


class InferenceTFMonteCarlo:
    """
    Inference with uncertainty
    """
    def __init__(self,
            dataprocessor: DataProcessorTF,
            model: tfk.Model,
            set_type: str,
            batch_size: int=64,
            unc_threshold: float=0.4,
            num_samples: int=5,
            overconfidence_analysis: bool=False,
            model_dir: Path=Path('.')
            ) -> None:
        self.dataset = dataprocessor.pad_dataset(batch_size)
        self.dataprocessor = dataprocessor
        self.model = model
        self.num_samples = num_samples
        self.unc_threshold = unc_threshold
        self.overconfidence_analysis = overconfidence_analysis
        self.model_dir = model_dir
        self.type = 'MC'
        self.set_type = set_type

    def run(self, chosen_case: str=""):
        """ Runs one epoch of the experiment """
        
        resultsprocessor = ResultProcessor(self.unc_threshold, chosen_case)

        for _, batch_data in enumerate(tqdm(self.dataset, desc='Epoch', position=1, leave=False)):
            input_data, target_data, target_data_case = \
                    self.dataprocessor.get_input_and_target(batch_data)

            out_prob_tot_distr, out_prob, u_t = self.predict(input_data)
            resultsprocessor.process_batch(
                    target_data, target_data_case, out_prob, out_prob_tot_distr, u_t, input_data)
        resultsprocessor.process_after_batches()
        if self.overconfidence_analysis:
            cases_id_train = compute_train_ids(self.dataprocessor.dataset_name)
            event_log = import_event_log(self.dataprocessor.dataset_name)
            features = {'features-type': self.dataprocessor.features_type, 
                        'features-variant': self.dataprocessor.features_variant}
            resultsprocessor.overconfidence_analysis(cases_id_train, event_log, features, self.dataprocessor.dataset_name)
            resultsprocessor.write_overconfidence_analysis_results(self.model_dir)
        self.unc_results = resultsprocessor.get_uncertainty_results()
        self.prob_results = resultsprocessor.get_probabilities_results()
        self.target_results = resultsprocessor.get_target_results()
        self.reliability_diagram = resultsprocessor.get_reliability_diagram_results()
        self.accuracy = resultsprocessor.get_accuracy_results()
        resultsprocessor.write_top_k_accuracy_results(self.model_dir, self.type)
        resultsprocessor.write_case_selected_numpy(self.model_dir, self.set_type.split("")[0])
        self.predict_case = resultsprocessor.get_predict_case_results()

    def compute_distributions(self, input_data: tf.Tensor) -> tf.Tensor:
        for sample in range(self.num_samples):
            logits = self.model(input_data, training=True)
            out_prob = tf.nn.softmax(logits)
            if  sample == 0:
                out_prob_tot_distr = copy(out_prob)
            else:
                out_prob_tot_distr += out_prob
        out_prob_tot_distr /= self.num_samples
        return out_prob_tot_distr, out_prob

    def predict(self, input_data):
        out_prob_tot_distr, out_prob = self.compute_distributions(input_data)
        u_t = -np.sum(out_prob_tot_distr.numpy() * np.log(out_prob_tot_distr.numpy()), axis=-1)
        return out_prob_tot_distr, out_prob, u_t

    def get_results(self):
        return {'uncertainty': self.unc_results, 'probability': self.prob_results, 'target': self.target_results,
                'reliability': self.reliability_diagram, 'accuracy': self.accuracy, 'case': self.predict_case}


class InferenceTFEnsemble:
    """
    Inference with uncertainty
    """
    def __init__(self,
            dataprocessor: DataProcessorTF,
            ensemble: list[tfk.Model],
            set_type: str,
            batch_size: int=64,
            unc_threshold: float=0.4,
            num_samples: int=5,
            overconfidence_analysis: bool=False,
            model_dir: Path=Path('.')
            ) -> None:
        self.dataset = dataprocessor.pad_dataset(batch_size)
        self.dataprocessor = dataprocessor
        self.ensemble = ensemble
        self.num_samples = num_samples
        self.unc_threshold = unc_threshold
        self.overconfidence_analysis = overconfidence_analysis
        self.model_dir = model_dir
        self.type = 'Ensemble'
        self.set_type = set_type

    def run(self, chosen_case: str=""):
        """ Runs one epoch of the experiment """
        
        resultsprocessor = ResultProcessor(self.unc_threshold, chosen_case)

        for _, batch_data in enumerate(tqdm(self.dataset, desc='Epoch', position=1, leave=False)):
            input_data, target_data, target_data_case = \
                    self.dataprocessor.get_input_and_target(batch_data)

            out_prob_tot_distr, out_prob, u_t = self.predict(input_data)
            resultsprocessor.process_batch(
                    target_data, target_data_case, out_prob, out_prob_tot_distr, u_t, input_data)

        resultsprocessor.process_after_batches()
        if self.overconfidence_analysis:
            cases_id_train = compute_train_ids(self.dataprocessor.dataset_name)
            event_log = import_event_log(self.dataprocessor.dataset_name)
            resultsprocessor.overconfidence_analysis(cases_id_train, event_log, self.dataprocessor.features, self.dataprocessor.dataset_name)
            resultsprocessor.write_overconfidence_analysis_results(self.model_dir)
        self.unc_results = resultsprocessor.get_uncertainty_results()
        self.prob_results = resultsprocessor.get_probabilities_results()
        self.target_results = resultsprocessor.get_target_results()
        self.reliability_diagram = resultsprocessor.get_reliability_diagram_results()
        self.accuracy = resultsprocessor.get_accuracy_results()
        resultsprocessor.write_top_k_accuracy_results(self.model_dir, self.type)
        resultsprocessor.write_case_selected_numpy(self.model_dir, self.set_type.split(" ")[0])
        self.predict_case = resultsprocessor.get_predict_case_results()

    def compute_distributions(self, input_data: tf.Tensor) -> tf.Tensor:
        for cont, model in enumerate(self.ensemble):
            logits = model(input_data, training=True)
            out_prob = tf.nn.softmax(logits)
            if  cont == 0:
                out_prob_tot_distr = copy(out_prob)
            else:
                out_prob_tot_distr += out_prob
        out_prob_tot_distr /= len(self.ensemble)
        return out_prob_tot_distr, out_prob

    def predict(self, input_data):
        out_prob_tot_distr, out_prob = self.compute_distributions(input_data)
        u_t = -np.sum(out_prob_tot_distr.numpy() * np.log(out_prob_tot_distr.numpy()), axis=-1)
        return out_prob_tot_distr, out_prob, u_t

    def get_results(self):
        return {'uncertainty': self.unc_results, 'probability': self.prob_results, 'target': self.target_results,
                'reliability': self.reliability_diagram, 'accuracy': self.accuracy, 'case': self.predict_case}

