import tensorflow as tf
import math
import itertools
import random
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten
import plotly.graph_objects as go
import logging


class EmbeddingModel:
    def __init__(self, df, type_model, initial_emb_path=None, num_units=20, emb_dim_mul=1):
        self._df = df
        self._emb_dim_mul = emb_dim_mul
        self._create_model(num_units, type_model, initial_emb_path)
        # variables needed for the training and plotting
        act_role_idx_df = self._df.loc[self._df['activity-role-index'].drop_duplicates().index, ['activity-role-index', 'activity-index', 'role']]
        # [activity-role, activity, role]
        self.act_role_idx = np.zeros((len(self._df['activity-role-index'].unique()) + 3, 3), dtype=int)
        self.act_role_idx[1, 0] = 1
        self.act_role_idx[2, 0] = 2
        self.act_role_idx[3:, 0] = act_role_idx_df['activity-role-index'].values
        self.act_role_idx[1, 1] = 1
        self.act_role_idx[2, 1] = 2
        self.act_role_idx[3:, 1] = act_role_idx_df['activity-index'].values
        self.act_role_idx[3:, 2] = act_role_idx_df['role'].values

    def _create_model(self, num_units, type_model, initial_emb_path):
        # create the compiled model for embedding
        # adding 3 for <START> <END> <PAD> 
        size_voc = len(self._df['activity-role-index'].unique()) + 3
        # number of output categories, add 1 for the 0 category reserved
        num_out_act = len(self._df['activity-index'].unique()) + 3
        num_out_role = len(self._df['role'].unique()) + 1
        # define number of embedded dimension 4th root of # of vocabulary
        self.emb_dim = math.ceil(size_voc**0.25)*self._emb_dim_mul
        self.type_model = type_model

        if type_model == 1:
            self.model = tf.keras.Sequential([
              tf.keras.layers.Embedding(size_voc, self.emb_dim, name='activity_role_embedding'),
              tf.keras.layers.Dense(num_units, activation='relu'),
              tf.keras.layers.Dense(num_out_role)])
            self.model.compile(optimizer='Adam',
                    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        elif type_model == 2:
            inputs = tf.keras.Input(shape=(1,))
            emb = tf.keras.layers.Embedding(size_voc, self.emb_dim, name='activity_role_embedding')(inputs)
            dense = tf.keras.layers.Dense(num_units, activation='relu')(emb)
            out_act = tf.keras.layers.Dense(num_out_act, name="activity_output")(dense)
            out_role = tf.keras.layers.Dense(num_out_role, name="role_output")(dense)
            self.model = tf.keras.Model(inputs=inputs, outputs=[out_act, out_role])
        
            losses = {"activity_output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    "role_output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
            loss_weights = {"activity_output": 1.0, "role_output": 1.0}

            self.model.compile(optimizer='Adam',
                    loss=losses, loss_weights=loss_weights,
                    metrics=['accuracy'])
        elif type_model == 3:
            initial_emb_path = os.path.join(initial_emb_path, 'activity-role-embedding-{}dim.npy'.format(self.emb_dim))
            self.model = tf.keras.Sequential([
              tf.keras.layers.Embedding(size_voc, self.emb_dim, weights=[np.load(initial_emb_path)],
                                        trainable=True, name='activity_role_embedding'),
              tf.keras.layers.Dense(num_units, activation='relu'),
              tf.keras.layers.Dense(num_out_act)])
            self.model.compile(optimizer='Adam',
                    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    def _generate_data(self, batch_size):
        # generator of batches
        if self.type_model == 1: 
            while True:
                rand_idx = np.random.randint(low=0, high=self.act_role_idx.shape[0], size=batch_size)
                yield self.act_role_idx[rand_idx, 0], self.act_role_idx[rand_idx, 2]
        elif self.type_model == 2:
            while True:
                rand_idx = np.random.randint(low=0, high=self.act_role_idx.shape[0], size=batch_size)
                yield self.act_role_idx[rand_idx, 0], [self.act_role_idx[rand_idx, 1], self.act_role_idx[rand_idx, 2]]
        elif self.type_model == 3: 
            while True:
                rand_idx = np.random.randint(low=0, high=self.act_role_idx.shape[0], size=batch_size)
                yield self.act_role_idx[rand_idx, 0], self.act_role_idx[rand_idx, 1]

    def _save_weights(self, dir_path):
        file_path = os.path.join(dir_path, 'activity-role-embedding-{}dim.npy'.format(self.emb_dim)) 
        np.save(file_path, 
                self.model.get_layer('activity_role_embedding').get_weights()[0])
        old_mask = os.umask(000)
        os.chmod(file_path, 0o777)
        _ = os.umask(old_mask)

    def train_embedding(self, save_dir_path, batch_size, epochs=100, patience=10, negative_ratio=1):
        generator = self._generate_data(batch_size)
        # compute number of events including starts and ends
        num_cases = len(self._df) + 2*len(self._df['case-id'].unique())
        self.model.fit(generator, epochs=epochs, 
                        steps_per_epoch=num_cases//batch_size,
                        shuffle=True,
                        verbose=2)
        self._save_weights(save_dir_path)

    def visualize_embedding(self, emb):
        fig = go.Figure(data=[
            go.Scatter3d(x=emb[:, 0], y=emb[:, 1], z=emb[:, 2], 
                mode='markers', marker=dict(color=self.act_role_idx[:, 1]))])
        fig.show()
