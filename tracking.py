from pathlib import Path
import tensorflow as tf
from dataprocessing import Stage


class ExperimentTracker:
    def __init__(self, stage: Stage, log_dir: Path):
        self.stage = stage
        self.log_dir = f'./{log_dir}'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.loss = tf.keras.metrics.Mean(name=f'{self.stage}_loss')
        self.accuracy = tf.keras.metrics.Mean(name=f'{self.stage}_accuracy')

    def track_loss(self, loss_value: tf.Tensor, step: int):
        self.loss(loss_value)
        with self.summary_writer.as_default():
            tf.summary.scalar('loss', self.loss.result(), step=step)

    def track_accuracy(self, accuracy_value: tf.Tensor, step: int):
        self.accuracy(accuracy_value)
        with self.summary_writer.as_default():
            tf.summary.scalar('accuracy', self.accuracy.result(), step=step)
