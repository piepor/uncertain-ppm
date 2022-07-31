from abc import ABC, abstractmethod
from typing import Any
from enum import Enum, auto
import tensorflow as tf
import tensorflow_datasets as tfds


class Stage(Enum):
    TRAIN = auto()
    VALI = auto()
    TEST = auto()

class DataProcessor(ABC):
    @abstractmethod
    def get_input_and_target(self, input_data: Any) -> tf.Tensor:
        pass


class DataProcessorTF(DataProcessor):
    def __init__(self, dataset_name: str, split: str,
            stage: Stage, dataset_map_utils: dict,
            features: list):
        self.shuffle = stage == Stage.TRAIN
        self.dataset = tfds.load(dataset_name, split=split, shuffle_files=self.shuffle)
        self.utils = dataset_map_utils[dataset_name] # TODO use config instead
        self.tfds_id = dataset_map_utils['tfds_id']
        self.features = features
        self.get_dataset_utils()

    def get_dataset_utils(self):
        padded_shapes, padding_values, vocabulary_act, _, _ = self.utils(self.tfds_id)
        self.padded_shapes = padded_shapes
        self.padding_values = padding_values
        self.vocabulary_act = vocabulary_act
        self.output_preprocess = tf.keras.layers.StringLookup(
                vocabulary=vocabulary_act, num_oov_indices=1)

    def pad_dataset(self, batch_size: int) -> tf.data.Dataset:
        padded_ds = self.dataset.padded_batch(
                batch_size,
                padded_shapes=self.padded_shapes,
                padding_values=self.padding_values)
        if self.shuffle:
            padded_ds.shuffle(
                    buffer_size=10000,
                    reshuffle_each_iteration=self.shuffle).prefetch(tf.data.AUTOTUNE)
        else:
            padded_ds.prefetch(tf.data.AUTOTUNE)

        return padded_ds

    def get_input_and_target(self, batch_data: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        input_data = []
        for feature in self.features:
            input_data.append(feature[:, :-1])
        target_data = self.output_preprocess(batch_data[0][0][:, 1:])
        return input_data, target_data
