from model_utils.py import Transformer
from training_utils import loss_function, accuracy_function
from model_utils import create_look_ahead_mask
import tensorflow as tf


class PredictiveModel:
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
                 pe_target, embedding_path, rate=0.1):
        self._model = Transformer(num_layers, d_model, num_heads, dff, 
                                  target_vocab_size, pe_target, embedding_path, rate=rate)
        self._train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
        self._train_loss = tf.keras.metrics.Mean(name='train_loss')
        self._train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self._val_loss = tf.keras.metrics.Mean(name='val_loss')
        self._val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def set_checkpoint(self, checkpoint_path, max_to_keep=1):
        self.ckpt = tf.train.Checkpoint(model=self._model, optimizer=self._optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

    @tf.function(input_signature=train_step_signature)
    def train_step(tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
    
        with tf.GradientTape() as tape:
            predictions, _ = self._model(tar_inp, True, look_ahead_mask)
            loss = loss_function(tar_real, predictions)
    
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
    
        self._train_loss(loss)
        self._train_accuracy(accuracy_function(tar_real, predictions))
    
    def val_step(tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
    
        predictions, _ = self._model(tar_inp, True, look_ahead_mask)
        loss = loss_function(tar_real, predictions)
        self._val_loss(loss)
        self._val_accuracy(accuracy_function(tar_real, predictions))
