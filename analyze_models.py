import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from preprocessing_utils import EventLogImporter, RolesDiscover, CreateSequences
from fastDamerauLevenshtein import damerauLevenshtein as DL
from model_utils import create_mask, positional_encodings
from model_types import Transformer
import os
import math
import numpy as np
import tensorflow_probability as tfp

def get_hash(x):
    return hash(tuple(x))

uncertainty = True

data_dir = './data'
model_dir = './models'

dataset = 'Helpdesk'
hash_code = '5579410409575014805'
res_data = pd.read_pickle('results/results-df.pkl')
mod_data = res_data.iloc[1]
path_df = os.path.join(data_dir, '{}.xes'.format(dataset))
hash_code = mod_data['hash']

#dataset = 'BPI-Challenge-2012'
#res_data = pd.read_pickle('results/results-df.pkl')
#mod_data = res_data.iloc[91]
#path_df = os.path.join(data_dir, '{}.xes'.format(dataset.replace('-', '_')))
##hash_code = '5579410409575014805'
#hash_code = mod_data['hash']

path_model = os.path.join(model_dir, dataset.lower(), 'type{}'.format(str(mod_data['embedding_type'])),
                          'predictive-model', hash_code, 'fold1')
#embedding_path = os.path.join(model_dir, dataset.lower(), 'type3', 'embedding', 'activity-role-embedding.npy')
embedding_path = None
index_train_path = os.path.join(model_dir, dataset.lower(), 'type{}'.format(str(mod_data['embedding_type'])), 
                                'predictive-model', hash_code,  'fold1', 'train_indeces.pkl') 
index_val_path = os.path.join(model_dir, dataset.lower(), 'type{}'.format(str(mod_data['embedding_type'])), 
                              'predictive-model', hash_code, 'fold1', 'val_indeces.pkl') 

NORMALIZATION = 'log-normal'
event_log = EventLogImporter(path_df, NORMALIZATION)
df = event_log.get_df()
role_discover = RolesDiscover(df)
df = role_discover.get_df()

# activities roles map
activities_role_map = df.drop_duplicates(
    subset='activity-role-index')[[
        'activity-role-index', 'activity-index', 'role']].set_index('activity-role-index').to_dict()
activities_map = activities_role_map['activity-index']
activities_map[0] = 0
activities_map[1] = 1
activities_map[2] = 2
roles_map = activities_role_map['role']
roles_map[0] = 0
roles_map[1] = 0
roles_map[2] = 0

train_indeces = pd.read_pickle(index_train_path)
vali_indeces = pd.read_pickle(index_val_path)
#breakpoint()
target_vocab_size = len(df['activity-role-index'].unique()) + 3
d_model = int(mod_data['d_model'])
PE = 1000
NUM_UNIT_DFF = int(mod_data['units'])
NUM_LAYERS = int(mod_data['layers'])
num_heads = int(mod_data['num_heads'])
TRAINABLE_EMB = True
predictive_model = Transformer(NUM_LAYERS, d_model, num_heads, NUM_UNIT_DFF,
        target_vocab_size, PE, embedding_path, TRAINABLE_EMB, rate=0.1)
start = np.ones((1, 1), dtype=int)
output = tf.convert_to_tensor(start)
mask = create_mask(output)
# predictions shape == (batch_size, seq_len, vocab_size)
predictions, attention_weights = predictive_model(output, False, mask)
predictive_model.load_weights(os.path.join(path_model, 'weights'))
#breakpoint()

# extract groups and compute hash
df_train = df[df['case-id'].isin(train_indeces)]
subdf_train = df_train[['case-id', 'activity-role-index', 'activity-index', 'role']]
grouped = subdf_train.groupby('case-id')
hash_case = grouped['activity-role-index'].apply(get_hash)
hash_case.name = 'hash'
subdf_train = pd.merge(subdf_train, hash_case, how='outer', on=['case-id'])

df_vali = df[df['case-id'].isin(vali_indeces)]
subdf_vali = df_vali[['case-id', 'activity-role-index', 'activity-index', 'role']]
grouped = subdf_vali.groupby('case-id')
hash_case = grouped['activity-role-index'].apply(get_hash)
hash_case.name = 'hash'
subdf_vali = pd.merge(subdf_vali, hash_case, how='outer', on=['case-id'])

#breakpoint()
# compute statistic
hash_count = subdf_train.drop_duplicates('case-id').pivot_table(index='hash', aggfunc='size')
hash_count.name = 'hash-count'
subdf_train = pd.merge(subdf_train, hash_count, how='outer', on=['hash'])
print(subdf_train.drop_duplicates('hash')['hash-count'].values.sum())
print(subdf_train.drop_duplicates('case-id').count())
#breakpoint()
#for gen_method in ['argmax', 'random']:
output_seq = tf.convert_to_tensor(start)
#print('------ {} ------'.format(gen_method.upper()))
method = 'random'
max_length_log = 50
beam_search_k = 20
alpha = 0.7 #0.7
k_top_seq = tf.convert_to_tensor(np.ones((beam_search_k, 1), dtype=np.int32))
softmax_for_score = tf.convert_to_tensor(np.zeros((beam_search_k, 1), dtype=np.float32))
scores = []
mask_top_k = tf.cast(tf.eye(beam_search_k, beam_search_k), tf.bool)
mask_top_k = tf.expand_dims(mask_top_k, axis=1)
#for i in range(max_length_log):
num_samples = 10
num_ended_seq = 0
i = 0
mask_end_seq = tf.cast(tf.zeros((beam_search_k, 1)), tf.bool)
#breakpoint()
if method == 'argmax':
    start = np.ones((1, 1), dtype=int)
    out_seq = tf.convert_to_tensor(start)
    predicted_id = 0
    while predicted_id != 2:
        mask = create_mask(out_seq)
        # predictions shape == (batch_size, seq_len, vocab_size)
        if uncertainty:
            predictions, attention_weights = predictive_model(out_seq, True, mask)
            pred_samples = np.zeros((num_samples, predictions.shape[0], predictions.shape[2]), dtype=np.float32)
            pred_samples[0, :, :] = predictions[:, -1, :].numpy()
            for sample in range(1, num_samples):
                predictions, attention_weights = predictive_model(out_seq, True, mask)
                pred_samples[sample, :, :] = predictions[:, -1, :].numpy()
            #breakpoint()
            predictions = tf.convert_to_tensor(np.median(pred_samples, axis=0))
            predictions = tf.expand_dims(predictions, axis=1)
        else:
            predictions, attention_weights = predictive_model(k_top_seq, False, mask)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
        # select the last word from the seq_len dimension
        predicted_id = tf.argmax(predictions, axis=2)
        out_seq = tf.concat([out_seq, predicted_id], axis=-1)
    print(out_seq.numpy().tolist())
elif method == 'beam':
    while num_ended_seq != beam_search_k and i < PE: 
        #breakpoint()
        if i > 0:
            # extract ended sequence and pad with zeros
            ended_seq = tf.boolean_mask(k_top_seq, tf.cast(mask_end_seq, tf.bool))
            ended_seq = tf.concat([ended_seq, tf.zeros((ended_seq.shape[0], 1), dtype=tf.int32)], axis=-1)
            # same for the scores
            softmax_for_score_ended = tf.boolean_mask(softmax_for_score, tf.cast(mask_end_seq, tf.bool))
            softmax_for_score_ended = tf.concat([softmax_for_score_ended, tf.zeros((softmax_for_score_ended.shape[0], 1))], axis=-1)
            mask_end_seq = tf.cast(tf.math.logical_not(tf.cast(mask_end_seq, tf.bool)), tf.float32)
            # compute new values only for ongoing case
            k_top_seq = tf.boolean_mask(k_top_seq, tf.cast(mask_end_seq, tf.bool))
            if tf.rank(k_top_seq) == 1:
                k_top_seq = tf.expand_dims(k_top_seq, axis=0)
            softmax_for_score = tf.boolean_mask(softmax_for_score, tf.cast(mask_end_seq, tf.bool))

        mask = create_mask(k_top_seq)
        if uncertainty:
            predictions, attention_weights = predictive_model(k_top_seq, True, mask)
            pred_samples = np.zeros((num_samples, predictions.shape[0], predictions.shape[2]), dtype=np.float32)
            pred_samples[0, :, :] = predictions[:, -1, :].numpy()
            for sample in range(1, num_samples):
                predictions, attention_weights = predictive_model(k_top_seq, True, mask)
                pred_samples[sample, :, :] = predictions[:, -1, :].numpy()
            #breakpoint()
            predictions = tf.convert_to_tensor(np.median(pred_samples, axis=0))
            predictions = tf.expand_dims(predictions, axis=1)
        else:
            predictions, attention_weights = predictive_model(k_top_seq, False, mask)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
        #breakpoint()
        predictions_top_k = tf.math.top_k(tf.nn.softmax(predictions), k=beam_search_k)
        if i == 0: 
            predictions_top_k_prob = tf.boolean_mask(predictions_top_k.values, mask_top_k)
            predictions_top_k_idx = tf.boolean_mask(predictions_top_k.indices, mask_top_k)
            reshape_shape = k_top_seq.shape[0]
        else:
            predictions_top_k_prob = predictions_top_k.values
            predictions_top_k_idx = predictions_top_k.indices
            reshape_shape = k_top_seq.shape[0]*beam_search_k
            softmax_for_score = tf.repeat(softmax_for_score, beam_search_k, axis=0)
            k_top_seq = tf.repeat(k_top_seq, beam_search_k, axis=0)
        # compute score and divide for length alpha = 0.7 from wu et al
        #score = -tf.math.log(predictions_top_k_prob) / tf.math.pow(tf.cast(k_top_seq.shape[0], tf.float32)+1, alpha)
        score = tf.math.log(predictions_top_k_prob) / tf.math.pow(tf.cast(k_top_seq.shape[0], tf.float32)+1, alpha)
        score = tf.reshape(score, shape=(reshape_shape, 1))
        softmax_for_score = tf.concat([softmax_for_score, score], axis=-1)
        linearized_predictions_idx = tf.reshape(predictions_top_k_idx, shape=(reshape_shape, 1))
        k_top_seq = tf.concat([k_top_seq, linearized_predictions_idx], axis=-1)
        if i > 0:
            # reinserting the ended sequences
            softmax_for_score = tf.concat([softmax_for_score_ended, softmax_for_score], axis=0)
            k_top_seq = tf.concat([ended_seq, k_top_seq], axis=0)
        # compute score 
        # using -softmax_for_score because topk get the largets eleements and -log(1) = 0 and -log(0.1) = 2.3 
        # -> most probable have low score
        top_k_score = tf.math.top_k(tf.math.reduce_sum(softmax_for_score, axis=1), k=beam_search_k)
        softmax_for_score = tf.gather(softmax_for_score, top_k_score.indices)
        # select words with overall best score 
        k_top_seq = tf.gather(k_top_seq, top_k_score.indices)
        #breakpoint()
        # compute mask for already ended sequences before repeating k_top_seq
        mask_end_seq = tf.reduce_sum(tf.cast(k_top_seq == 2, tf.float32), axis=1)
        num_ended_seq = tf.reduce_sum(mask_end_seq)
        i += 1
        # concat to the previous sequence
        #k_top_seq = tf.concat([k_top_seq, predicted_id], axis=-1)
# to numpy
    print(k_top_seq.numpy().tolist())
elif method == 'random':
    start = np.ones((beam_search_k, 1), dtype=np.int32)
    out_seq = tf.convert_to_tensor(start)
    total_ended_seq = []
    while num_ended_seq != beam_search_k and i < PE: 
        #breakpoint()
        if i > 0:
            # extract ended sequence and pad with zeros
            ended_seq = tf.boolean_mask(out_seq, tf.cast(mask_end_seq, tf.bool))
            ended_seq = tf.concat([ended_seq, tf.zeros((ended_seq.shape[0], 1), dtype=tf.int32)], axis=-1)
            # same for the scores
            mask_end_seq = tf.cast(tf.math.logical_not(tf.cast(mask_end_seq, tf.bool)), tf.float32)
            # compute new values only for ongoing case
            out_seq = tf.boolean_mask(out_seq, tf.cast(mask_end_seq, tf.bool))

        mask = create_mask(out_seq)
        if uncertainty:
            predictions, attention_weights = predictive_model(out_seq, True, mask)
            pred_samples = np.zeros((num_samples, predictions.shape[0], predictions.shape[2]), dtype=np.float32)
            pred_samples[0, :, :] = predictions[:, -1, :].numpy()
            for sample in range(1, num_samples):
                predictions, attention_weights = predictive_model(out_seq, True, mask)
                pred_samples[sample, :, :] = predictions[:, -1, :].numpy()
            #breakpoint()
            predictions = tf.convert_to_tensor(np.median(pred_samples, axis=0))
            predictions = tf.expand_dims(predictions, axis=1)
        else:
            predictions, attention_weights = predictive_model(out_seq, False, mask)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
        #breakpoint()
        predicted_id = tf.random.categorical(predictions[:, 0, :], 1, dtype=tf.int32)
        out_seq = tf.concat([out_seq, predicted_id], axis=-1)
        if i > 0:
            # reinserting the ended sequences
            out_seq = tf.concat([ended_seq, out_seq], axis=0)
        # compute mask for already ended sequences before repeating k_top_seq
        mask_end_seq = tf.reduce_sum(tf.cast(out_seq == 2, tf.float32), axis=1)
        num_ended_seq = tf.reduce_sum(mask_end_seq)
        i += 1
        #breakpoint()
    print(out_seq.numpy().tolist())

if method == 'beam':
    final_sequences = k_top_seq.numpy().tolist()
else:
    final_sequences = out_seq.numpy().tolist()

count_hash_gen_seq = np.zeros((len(final_sequences,)))
for i, seq in enumerate(final_sequences):
    if 2 in seq:
        end_idx = seq.index(2)
        cut_seq = seq[1:end_idx]
        hash_seq = hash(tuple(cut_seq))
        if hash_seq in subdf_train['hash'].values:
            count_hash_gen_seq[i] = subdf_train.loc[subdf_train['hash']==hash_seq].drop_duplicates('hash')['hash-count'].values[0]

perc_gen_seq = count_hash_gen_seq / len(subdf_train['case-id'].unique())
print(perc_gen_seq)

