import pandas as pd
from fastDamerauLevenshtein import damerauLevenshtein as DL
from tqdm import tqdm
import tensorflow as tf
from preprocessing_utils import EventLogImporter, RolesDiscover, CreateSequences
from model_utils import create_mask, positional_encodings
from model_types import Transformer
import os
import math
import numpy as np
from itertools import compress

def compute_dl(actual_series, predicted_series):
    #predicted_series = predicted_series.numpy().tolist()
    idx = predicted_series.index(2)
    predicted_series = predicted_series[0:idx+1]
    actual_series = list(map(lambda x: activities_map[x], actual_series))
    predicted_series = list(map(lambda x: activities_map[x], predicted_series))
    return DL(actual_series, predicted_series)

# interest case validation 'Case4567'
dataset = 'Helpdesk'
#dataset = 'BPI_Challenge_2012'
data_dir = './data'
model_dir = './models'
res_data = pd.read_pickle('results/results-df.pkl')
#mod_data = res_data.iloc[1]
mod_data = res_data.iloc[116]
#mod_data = res_data.iloc[115]
dataset_name = mod_data['dataset']
path_df = os.path.join(data_dir, '{}.xes'.format(dataset))
hash_code = mod_data['hash']
path_model = os.path.join(model_dir, dataset_name, 'type{}'.format(str(mod_data['embedding_type'])),
                          'predictive-model', hash_code, 'fold1')
#embedding_path = os.path.join(model_dir, dataset.lower(), 'type3', 'embedding', 'activity-role-embedding.npy')
embedding_path = None
index_train_path = os.path.join(model_dir, dataset_name, 'type{}'.format(str(mod_data['embedding_type'])), 
                                'predictive-model', hash_code,  'fold1', 'train_indeces.pkl') 
index_val_path = os.path.join(model_dir, dataset_name, 'type{}'.format(str(mod_data['embedding_type'])), 
                              'predictive-model', hash_code, 'fold1', 'val_indeces.pkl') 
BATCH_SIZE = 128

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

# create static look up table for tensorflow for activities
table = tf.lookup.StaticVocabularyTable(
    tf.lookup.KeyValueTensorInitializer(list(activities_map.keys()), list(activities_map.values()),
                                        key_dtype=tf.int64, value_dtype=tf.int64), num_oov_buckets=1)

#breakpoint()
train_indeces = pd.read_pickle(index_train_path)
vali_indeces = pd.read_pickle(index_val_path)
df_train = df[df['case-id'].isin(train_indeces)]
df_vali = df[df['case-id'].isin(vali_indeces)]

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

df_series = df_vali.groupby('case-id')['activity-role-index'].agg(
    lambda x: list(x)).values.tolist()
tot_actual_series = []
for i in range(len(df_series)):
    seq = [1]
    seq.extend(df_series[i])
    seq.extend([2])
    tot_actual_series.append(seq)

cont_batch = 0
#while cont_batch > len(tot_actual_series):
#actual_series = tot_actual_series[0:10]
#actual_series = tot_actual_series
actual_series = tot_actual_series[cont_batch:cont_batch+BATCH_SIZE]
original_len_actual_series = len(actual_series)
initial_input = list(map(lambda x: x[0], actual_series))
initial_input = np.asarray(initial_input)
initial_input = initial_input[:, np.newaxis]
gen_series = tf.convert_to_tensor(initial_input)
cont_input_seq = 0
gen_finished = [True]*len(gen_series)
finished = [False]*len(gen_series)
average_dl_suffix = []
print("Number of sequence: {}".format(len(tot_actual_series)))
while cont_batch < len(tot_actual_series):
    print(cont_batch)
    while actual_series:
        #breakpoint()
        while finished != gen_finished:
            #breakpoint()
            mask = create_mask(gen_series)
            predictions, attention_weights = predictive_model(gen_series, False, mask)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.random.categorical(predictions[:, 0, :], 1)
            #predicted_id = table.lookup(predicted_id)
            #pred_list = list(map( lambda x: activities_map[], predicted_id.numpy().tolist()))
            # concat to the previous sequence
            gen_series = tf.concat([gen_series, predicted_id], axis=-1)
            # check what sequences has throw an <END>
            new_finished = list(map(lambda x: x[-1].numpy().tolist() == 2, gen_series))
            old_finished = finished
            finished = [bool1 or bool2 for bool1, bool2 in zip(new_finished, old_finished)]
            #breakpoint()
        # compute dl
        #breakpoint()
        actual_series_for_dl = list(map( lambda x: x[cont_input_seq+1:], actual_series))
        gen_series_for_dl = list(map( lambda x: x[cont_input_seq+1:], gen_series.numpy().tolist()))
        
        #breakpoint()
        dl_suffix = list(map(compute_dl, actual_series_for_dl, gen_series_for_dl))
        average_dl_suffix.append(sum(dl_suffix)/len(dl_suffix))
        # check what actual series has ended
        actual_series_not_finished = list(map(
            lambda x: x[cont_input_seq+1] != 2, actual_series))
        actual_series = list(compress(actual_series, actual_series_not_finished))
        if actual_series:
            cont_input_seq += 1
            gen_series = np.asarray(list(map(lambda x: x[0:cont_input_seq+1], actual_series)))
            gen_series = tf.convert_to_tensor(gen_series)
            finished = [False]*len(gen_series)
            gen_finished = [True]*len(gen_series)
    #breakpoint()
    if len(tot_actual_series) - cont_batch < BATCH_SIZE:
        actual_series = tot_actual_series[cont_batch:]
        cont_batch += BATCH_SIZE
    else:
        cont_batch += BATCH_SIZE
        actual_series = tot_actual_series[cont_batch:cont_batch+BATCH_SIZE]
    initial_input = list(map(lambda x: x[0], actual_series))
    initial_input = np.asarray(initial_input)
    initial_input = initial_input[:, np.newaxis]
    gen_series = tf.convert_to_tensor(initial_input)
    finished = [False]*len(gen_series)
    gen_finished = [True]*len(gen_series)
    cont_input_seq = 0

print("Mean dl suffix: {}".format(sum(average_dl_suffix)/len(average_dl_suffix)))


#while finished != all_finished:
#    #input_seq_tens = extend_actual_input(input_seq_tens, df_vali r)
#    while finished_generation != all_finished_generation:
#        mask = create_mask(input_seq_tens)
#        predictions, attention_weights = predictive_model(input_seq_tens, False, mask)
