import tensorflow as tf
import tensorflow.keras as tfk
from abc import ABC, abstractmethod
from utils import accuracy_function 
from tqdm import tqdm


class ExperimentRunner(ABC):
    """
    Responsible of running the machine learning experiment or training. 
    """
    @abstractmethod
    def run(self, descr: str, experiment: ExperimentTracker):
        """ Implements running one experiment epoch """
        pass

class ExperimentRunnerTF(ExperimentRunner):
    """
    Implementation of the experiment using the Tensorflow library
    """
    def __init__(self, 
            dataprocessor: DataProcessor,
            model: tfk.Model,
            optimizer: Optional[tfk.optimizer] = None,
            loss_function: tfk.losses.Loss,
            ) -> None:
        self.dataset = dataprocessor.dataset
        self.dataprocessor = dataprocessor
        self.model = model
        self.optimizer = optimizer
        self.loss = loss_function
        self.accuracy_function = accuracy_function
        
    def run(self, desc: str, experiment: ExperimentTracker, epochs: int):
        """ Runs one epoch of the experiment """

        for step, batch_data in enumerate(tqdm(self.dataset, desc='Epoch', position=1, leave=False)):
            input_data, target_data = self.dataprocessor.get_input_and_target(batch_data)
            logits, loss_value = self._one_step(input_data, target_data)
            experiment.rec_loss(loss_value)
            experiment.rec_accuracy(self.accuracy_function(target_data, logits))

    def _one_step(self, input_data: Any, target_data: Any) -> [tf.Tensor, tf.Tensor]:
        """ Execute one step of the epoch. Backpropagates only if is present the optimizer """

        with tf.GradientTape() as tape:
            logits = self.model(input_data, training=True) 
            loss_value = self.loss_function(target_data, logits)
            if optimizer:
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return logits, loss_value
