from abc import ABC, abstractmethod
from typing import Any
from enum import Enum, auto
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from utilities.generals import compute_features


class Mode(Enum):
    TRAIN = auto()
    VISUALIZE = auto()


class Stage(Enum):
    TRAIN = auto()
    VALI = auto()
    TEST = auto()

class DataProcessor(ABC):
    @abstractmethod
    def get_input_and_target(self, input_data: Any) -> tf.Tensor:
        """ get input and target from a batch of data """


class DataProcessorTF(DataProcessor):
    def __init__(self, dataset_name: str, stage: Stage, dataset_map_utils: dict, tfds_id: bool=False):
        self.dataset_name = dataset_name
        self.dataset = None
        self.shuffle = stage == Stage.TRAIN
        self.stage = stage
        self.utils = dataset_map_utils[dataset_name] # TODO use config instead
        self.tfds_id = tfds_id
        self.get_dataset_utils()

    def import_dataset_inference(self, split_map):
        read_config = tfds.ReadConfig()
        read_config.add_tfds_id = True
        builder_ds = tfds.builder(self.dataset_name)
        self.dataset = builder_ds.as_dataset(read_config=read_config, split=split_map[self.stage])

    def import_dataset(self, split_map):
        self.dataset = tfds.load(self.dataset_name, split=split_map[self.stage], shuffle_files=self.shuffle)

    def get_dataset_utils(self):
        padded_shapes, padding_values, vocabularies, features_type, features_variant = self.utils(self.tfds_id)
        self.padded_shapes = padded_shapes
        self.padding_values = padding_values
        self.vocabularies = vocabularies
        self.output_preprocess = tf.keras.layers.StringLookup(
                vocabulary=self.vocabularies['activity'], num_oov_indices=1)
        self.features_type = features_type
        self.features_variant = features_variant

    def compute_features(self, file_path):
        self.features = compute_features(file_path, self.vocabularies)

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
        input_data = []
        for feature in self.features:
            input_data.append(batch_data[feature['name']][:, :-1])
        target_data = batch_data['activity'][:, 1:]
        target_data_case = self.get_target_case(target_data, batch_data['case_id'])
        target_data = self.output_preprocess(target_data)
        return input_data, target_data, target_data_case

    def get_target_case(self, target_data: tf.Tensor, batch_case_id: tf.Tensor) -> np.array:
        exs = [ex.numpy()[0].decode('utf-8') for ex in batch_case_id]
        #target_data_case = [idx_to_int(tfds_id, builder_ds) for tfds_id in exs]
        target_data_case = [int(case_id) for case_id in exs]
        target_data_case = target_data_case * np.ones_like(target_data).T
        target_data_case = target_data_case.T
        return target_data_case
