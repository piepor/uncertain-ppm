import tensorflow_datasets as tfds
from abc import ABC, abstractmethod


class DataProcessor(ABC):
    @abstractmethod
    def get_input_and_target(self, input_data: Any) -> tf.Tensor:
        pass


class DataProcessorTF(DataProcessor):
    def __init__(dataset_name: str, split: str, stage: Stage, dataset_map_utils: dict):
        self.shuffle = stage == Stage.TRAIN
        self.dataset = tfds.load(dataset_name, split=split, shuffle_files=shuffle)
        self.utils = dataset_map_utils[dataset_name] # TODO use config instead
        self.tfds_id = dataset_map_utils['tfds_id']
        self.get_dataset_utils
        
    def get_dataset_utils(self):
        padded_shapes, padding_values, vocabulary_act, _, _ = self.utils(self.tfds_id)
        self.padded_shapes = padded_shapes
        self.padding_values = paddeing_values
        self.vocabulary_act = vocabulary_act
        self.output_preprocess = tf.keras.layers.StringLookup(
                vocabulary=vocabulary_act, num_oov_indices=1)

    def pad_dataset(self, batch_size: int) -> tf.data.Dataset:
        padded_ds = self.dataset.padded_batch(
                batch_size, 
                padded_shapes=padded_shapes,
                padding_values=padding_values)
        if self.shuffle:
            padded_ds.shuffle(
                    buffer_size=10000,
                    reshuffle_each_iteration=self.shuffle).prefetch(tf.data.AUTOTUNE)
        else:
            padded_ds.prefetch(tf.data.AUTOTUNE)

        return padded_ds

    def get_input_and_target(self, input_data: Any) -> tf.Tensor:
        return self.output_preprocess(input_data)


if dataset == 'helpdesk':
    ds_train = tfds.load(dataset, split='train[:70%]', shuffle_files=True)
    ds_vali = tfds.load(dataset, split='train[70%:85%]')
    ds_test = tfds.load(dataset, split='train[85%:]')
    padded_shapes, padding_values, vocabulary_act, _, _ = helpdesk_utils(tfds_id=False)
    features = compute_features(features_name, {'activity': vocabulary_act})
    output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary_act,
                                                     num_oov_indices=1)
elif dataset == 'bpic2012':
    ds_train = tfds.load(dataset, split='train[:70%]', shuffle_files=True)
    ds_vali = tfds.load(dataset, split='train[70%:85%]')
    ds_test = tfds.load(dataset, split='train[85%:]')
    padded_shapes, padding_values, vocabulary_act, vocabulary_res, _, _ = bpic2012_utils(tfds_id=False)
    features = compute_features(features_name, 
                                {'activity': vocabulary_act, 'resource': vocabulary_res})
    output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary_act,
                                                     num_oov_indices=1)

train_step_signature = compute_input_signature(features)
embed_dim = sum([feature['output-dim'] for feature in features])


