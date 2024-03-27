from typing import Optional
import tensorflow as tf
import tensorflow.keras as tfk
from abc import ABC, abstractmethod
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile
import numpy as np
from dataprocessing import DataProcessorTF, Stage
from tracking import ExperimentTracker
from utilities.generals import loss_function, accuracy_function, compute_input_signature
import training
from utilities.directories import create_models_dir, create_single_model_dirs
from utilities.model import GeneralModel


class ExperimentRunner(ABC):
    """
    Responsible of running the machine learning experiment or training.
    """
    @abstractmethod
    def run(self, experiment: ExperimentTracker, epoch: int):
        """ Implements running one experiment epoch """


class ExperimentRunnerTFOneModel(ExperimentRunner):
    """
    Implementation of the experiment using the Tensorflow library
    """
    def __init__(self,
                 model: tfk.Model,
                 loss_obj: tfk.losses.Loss,
                 optimizer: tfk.optimizers,
                 num_epochs: int,
                 patience: int,
                 batch_size: int,
                 ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss_obj
        self.accuracy_function = accuracy_function
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_size = batch_size

    def run(self, experiment: ExperimentTracker, epoch: int, training: bool, 
            dataprocessor: DataProcessorTF, padded_ds: tf.data.Dataset, desc: str='Epoch'):
        """ Runs one epoch of the experiment """

        for _, batch_data in enumerate(tqdm(padded_ds, desc=desc, position=1, leave=False)):
            input_data, target_data, _ = dataprocessor.get_input_and_target(batch_data)
            logits, loss_value = self._one_step(input_data, target_data, training)
            experiment.track_loss(loss_value, epoch)
            experiment.track_accuracy(self.accuracy_function(target_data, logits), epoch)

    # TODO: add input signature for speeding up training
    def _one_step(self, input_data: tf.Tensor, target_data: tf.Tensor, training: bool) -> [tf.Tensor, tf.Tensor]:
        """ Execute one step of the epoch. Backpropagates only if is present the optimizer """

        with tf.GradientTape() as tape:
            logits = self.model(input_data, training=True)
            loss_value = loss_function(target_data, logits, self.loss)
        if training:
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return logits, loss_value

    def train(self, train_tracker: ExperimentTracker, vali_tracker: ExperimentTracker,
              train_dataprocessor: DataProcessorTF, vali_dataprocessor: DataProcessorTF):
        """ Train the model """

        padded_ds_train = train_dataprocessor.pad_dataset(self.batch_size)
        padded_ds_vali = vali_dataprocessor.pad_dataset(self.batch_size)
        wait = 0 
        best = 0
        for epoch in tqdm(range(self.num_epochs), desc='Train', position=0):
            self.run(train_tracker, epoch, True, train_dataprocessor, padded_ds_train)
            self.run(vali_tracker, epoch, False, vali_dataprocessor, padded_ds_vali, 'Vali')
            # Patience method for the early stop mechanism
            wait += 1
            if vali_tracker.loss.result() < best or epoch == 0:
                best = vali_tracker.loss.result()
                wait = 0
            
            train_tracker.loss.reset_states()
            train_tracker.accuracy.reset_states()
            vali_tracker.loss.reset_states()
            vali_tracker.accuracy.reset_states()

            if wait >= self.patience:
                break


class ExperimentRunnerTFEnsemble(ExperimentRunner):
    """
    Implementation of the experiment using the Tensorflow library
    """
    def __init__(self, 
                 models_dir: Path,
                 train_dataprocessor: DataProcessorTF,
                 vali_dataprocessor: DataProcessorTF,
                 params: dict) -> None:
        self.models_dir = models_dir
        self.accuracy_function = accuracy_function
        self.train_dataprocessor = train_dataprocessor
        self.vali_dataprocessor = vali_dataprocessor
        self.params = params

    def get_ensemble(self):
        models_paths = [child for child in self.models_dir.iterdir() if child.is_dir()]
        models = []
        for model_path in models_paths:
            model = tfk.models.load_model(f"./{str(model_path)}")
            models.append(model)
        return models

    def validate_ensemble(self, params: dict):
        ensemble = self.get_ensemble()
        padded_ds = self.vali_dataprocessor.pad_dataset(self.params['batch-size'])
        final_acc = self.run_validate(padded_ds, ensemble)
        self.write_results(final_acc, params)

    def run_validate(self, padded_ds: tf.data.Dataset, models: list):
        final_acc = []
        for batch_data in tqdm(padded_ds, desc='Vali', position=0, leave=False):
            logits_total = []
            output_total = []
            target_data = []
            input_data, target_data, _ = self.vali_dataprocessor.get_input_and_target(batch_data)
            for n, model in enumerate(models):
                input_signature = compute_input_signature(self.vali_dataprocessor.features)
                @tf.function(input_signature=[input_signature])
                def vali_step(*args):
                    logits = model(args[0], training=False) 
                    return logits
                if n == 0:
                    logits_total = vali_step(input_data)
                    output_total = tf.nn.softmax(logits_total)
                else:
                    logits_single = vali_step(input_data)
                    logits_total += logits_single
                    output_total += tf.nn.softmax(logits_total)
            output_total /= len(models)
            final_acc.append(self.accuracy_function(target_data, output_total).numpy().mean())
        return final_acc

    def run(self, params):
        # train models of the ensemble
        for _ in range(params['num-ensemble']):
            model_path, train_log_dir, vali_log_dir = create_single_model_dirs(self.models_dir)
            tfk.backend.clear_session()
            optimizer = tfk.optimizers.Adam()
            loss_obj = tfk.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            training_tracker = ExperimentTracker(Stage.TRAIN, train_log_dir)
            validation_tracker = ExperimentTracker(Stage.VALI, vali_log_dir)
            #train loop
            model = GeneralModel(params['num-layers'], self.train_dataprocessor.features,
                                 self.train_dataprocessor.dataset, params['num-heads'], params['feed-forward-dim'])
            experiment_runner = training.ExperimentRunnerTFOneModel(
                    model, loss_obj, optimizer, params['num-epochs'], params['patience'], params['batch-size'])
            experiment_runner.train(training_tracker, validation_tracker, self.train_dataprocessor, self.vali_dataprocessor)
            model.save(model_path)
        self.validate_ensemble(params)

    def write_results(self, final_acc: list, params: dict):
        with open(self.models_dir / "model_properties_results.txt", "a") as file:
            file.write("Number of heads: {}\n".format(params['num-heads']))
            file.write("Feed forward dimension: {}\n".format(params['feed-forward-dim']))
            file.write("Number of layers: {}\n".format(params['num-layers']))
            file.write("Number of epochs: {}\n".format(params['num-epochs']))
            file.write("Epochs of patience: {}\n".format(params['patience']))
            file.write("Size of batches: {}\n".format(params['batch-size']))
            file.write("Number of ensemble models: {}\n".format(params['num-ensemble']))
            file.write("Final validation accuracy: {}\n".format(np.asarray(final_acc).mean()))
