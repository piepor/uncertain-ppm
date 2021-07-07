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
import plotly.graph_objects as go

def get_hash(x):
    return hash(tuple(x))

dataset = 'Helpdesk'
hash_code = '5579410409575014805'
res_data = pd.read_pickle('results/results-df.pkl')
mod_data = res_data.iloc[117]
data_dir = './data'
path_df = os.path.join(data_dir, '{}.xes'.format(dataset))
hash_code = mod_data['hash']
model_dir = './models'

#dataset = 'BPI-Challenge-2012'
#hash_code = '-3369571847896487186'
#model_dir = './models_old'
#data_dir = './data'
#res_data = pd.read_pickle('results/results-df-old2.pkl')
#mod_data = res_data.iloc[15]

#dataset = 'BPI-Challenge-2012'
#data_dir = './data'
#model_dir = './models'
#res_data = pd.read_pickle('results/results-df.pkl')
#mod_data = res_data.iloc[91]
#path_df = os.path.join(data_dir, '{}.xes'.format(dataset).replace('-', '_'))
#hash_code = mod_data['hash']

index_train_path = os.path.join(model_dir, dataset.lower(), 'type{}'.format(str(mod_data['embedding_type'])), 
                                'predictive-model', hash_code,  'fold1', 'train_indeces.pkl') 
index_val_path = os.path.join(model_dir, dataset.lower(), 'type{}'.format(str(mod_data['embedding_type'])), 
                              'predictive-model', hash_code, 'fold1', 'val_indeces.pkl') 
image_dir = "./images"
num_most_prob_seq = 10
num_seq_hist = 25

NORMALIZATION = 'log-normal'
event_log = EventLogImporter(path_df, NORMALIZATION)
df = event_log.get_df()
role_discover = RolesDiscover(df)
df = role_discover.get_df()
print('Analyzing {} dataset'.format(dataset))

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

hash_count = subdf_vali.drop_duplicates('case-id').pivot_table(index='hash', aggfunc='size')
hash_count.name = 'hash-count'
subdf_vali = pd.merge(subdf_vali, hash_count, how='outer', on=['hash'])
print(subdf_vali.drop_duplicates('hash')['hash-count'].values.sum())
print(subdf_vali.drop_duplicates('case-id').count())

df_tot = df[['case-id', 'activity-role-index', 'activity-index', 'role']]
grouped = df_tot.groupby('case-id')
hash_case = grouped['activity-role-index'].apply(get_hash)
hash_case.name = 'hash'
df_tot = pd.merge(df_tot, hash_case, how='outer', on=['case-id'])
hash_count = df_tot.drop_duplicates('case-id').pivot_table(index='hash', aggfunc='size')
hash_count.name = 'hash-count'
df_tot = pd.merge(df_tot, hash_count, how='outer', on=['hash'])
print(df_tot.drop_duplicates('hash')['hash-count'].values.sum())
print(df_tot.drop_duplicates('case-id').count())
#breakpoint()

# compute similarity in the actual training and validation dataset
# of the most repeated sequence
dl_act_best = 0
dl_role_best = 0
#breakpoint()
# get the most present 
max_n_seq_case = subdf_train.loc[subdf_train['hash-count'].max()]['case-id']
max_n_seq = [1]
max_n_seq.extend(subdf_train.loc[subdf_train['case-id']==max_n_seq_case, 
                                 'activity-role-index'].values)
max_n_seq.extend([2])
max_n_seq_act = list(map(lambda x: activities_map[x], max_n_seq))
max_n_seq_role = list(map(lambda x: roles_map[x], max_n_seq))
dl_act_total = 0
dl_role_total = 0
for case_hash in tqdm(subdf_train['hash'].unique()):
    case_name = subdf_train.loc[subdf_train['hash']==case_hash, 'case-id'].iloc[0]
    hash_count = len(subdf_train.loc[subdf_train['hash']==case_hash, 'case-id'].unique())
    seq_activity = [1]
    seq_activity.extend(subdf_train.loc[subdf_train['case-id']==case_name, 
                                        'activity-index'].values)
    seq_activity.extend([2])
    seq_role = [0]
    seq_role.extend(subdf_train.loc[subdf_train['case-id']==case_name, 'role'].values)
    seq_role.extend([0])
    dl_act = DL(seq_activity, max_n_seq_act)
    dl_role = DL(seq_role, max_n_seq_role)
    dl_act_total += dl_act*hash_count
    dl_role_total += dl_role*hash_count

dl_act_mean_train = dl_act_total / len(subdf_train['case-id'].unique())
dl_role_mean_train = dl_role_total / len(subdf_train['case-id'].unique())

dl_act_total = 0
dl_role_total = 0
for case_hash in tqdm(subdf_vali['hash'].unique()):
    case_name = subdf_vali.loc[subdf_vali['hash']==case_hash, 'case-id'].iloc[0]
    hash_count = len(subdf_vali.loc[subdf_vali['hash']==case_hash, 'case-id'].unique())
    seq_activity = [1]
    seq_activity.extend(subdf_vali.loc[subdf_vali['case-id']==case_name, 
                                        'activity-index'].values)
    seq_activity.extend([2])
    seq_role = [0]
    seq_role.extend(subdf_vali.loc[subdf_vali['case-id']==case_name, 'role'].values)
    seq_role.extend([0])
    dl_act = DL(seq_activity, max_n_seq_act)
    dl_role = DL(seq_role, max_n_seq_role)
    dl_act_total += dl_act*hash_count
    dl_role_total += dl_role*hash_count

dl_act_mean_vali = dl_act_total / len(subdf_vali['case-id'].unique())
dl_role_mean_vali = dl_role_total / len(subdf_vali['case-id'].unique())

print('-----------------')
print('DL activities on training set: {}'.format(dl_act_mean_train))
print('DL roles on training set: {}'.format(dl_role_mean_train))
print('-----------------')
print('DL activities on validation set: {}'.format(dl_act_mean_vali))
print('DL roles on validation set: {}'.format(dl_role_mean_vali))

# plot statistics on entire dataframe
#df_tot = subdf_train.append(subdf_vali)

dl_act_total = 0
dl_role_total = 0
for case_hash in tqdm(df_tot['hash'].unique()):
    case_name = df_tot.loc[df_tot['hash']==case_hash, 'case-id'].iloc[0]
    hash_count = len(df_tot.loc[df_tot['hash']==case_hash, 'case-id'].unique())
    seq_activity = [1]
    seq_activity.extend(df_tot.loc[df_tot['case-id']==case_name, 
                                        'activity-index'].values)
    seq_activity.extend([2])
    seq_role = [0]
    seq_role.extend(df_tot.loc[df_tot['case-id']==case_name, 'role'].values)
    seq_role.extend([0])
    dl_act = DL(seq_activity, max_n_seq_act)
    dl_role = DL(seq_role, max_n_seq_role)
    dl_act_total += dl_act*hash_count
    dl_role_total += dl_role*hash_count

dl_act_mean_total = dl_act_total / len(df_tot['case-id'].unique())
dl_role_mean_total = dl_role_total / len(df_tot['case-id'].unique())
print('-----------------')
print('DL activities on total set: {}'.format(dl_act_mean_train))
print('DL roles on total set: {}'.format(dl_role_mean_train))

# TRAINING SET
data_hist = []
seq_sort = subdf_train.drop_duplicates('case-id').groupby('hash').count().sort_values(
    ['case-id'], ascending=False)
hash_sort = seq_sort.index.values.tolist()
seq_number = seq_sort['case-id'].values.tolist()
cases_name_sort = []
#breakpoint()
for i, unique_seq in enumerate(seq_number[:num_seq_hist]):
    name_seq = subdf_train[subdf_train['hash']==hash_sort[i]]['case-id'].values[0]
    if not dataset == 'Helpdesk':
        name_seq = 'Case{}'.format(name_seq)
    cases_name_sort.append(name_seq)
    data_hist.extend(unique_seq*[name_seq])
fig = go.Figure()
fig.add_trace(go.Histogram(x=data_hist))
#fig.update_layout(title='Distribution of unique sequences in {} dataset'.format(dataset))
fig.show()
#fig.update_xaxes(showticklabels=False)
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=np.arange(1, num_seq_hist+1), y=seq_number[:num_seq_hist]))
fig_bar.write_image(os.path.join(image_dir, 'unique-seq-training-{}-{}.svg'.format(dataset, num_seq_hist)))
print("TRAINIG SET")
print("Percentage of default sequence: {:.4f}".format(seq_number[0]/len(subdf_train['case-id'].unique())))
print("Proportion of second most repeated sequence with respect to the default: {:.4f}".format(seq_number[1]/seq_number[0]))
print("Total number of traces: {}".format(len(subdf_train['case-id'].unique())))

# VALIDATION SET
data_hist = []
seq_sort = subdf_vali.drop_duplicates('case-id').groupby('hash').count().sort_values(
    ['case-id'], ascending=False)
hash_sort = seq_sort.index.values.tolist()
seq_number = seq_sort['case-id'].values.tolist()
cases_name_sort = []
#breakpoint()
for i, unique_seq in enumerate(seq_number[:num_seq_hist]):
    name_seq = subdf_vali[subdf_vali['hash']==hash_sort[i]]['case-id'].values[0]
    if not dataset == 'Helpdesk':
        name_seq = 'Case{}'.format(name_seq)
    cases_name_sort.append(name_seq)
    data_hist.extend(unique_seq*[name_seq])
fig = go.Figure()
fig.add_trace(go.Histogram(x=data_hist))
#fig.update_layout(title='Distribution of unique sequences in {} dataset'.format(dataset))
fig.show()
#fig.update_xaxes(showticklabels=False)
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=np.arange(1, num_seq_hist+1), y=seq_number[:num_seq_hist]))
fig_bar.write_image(os.path.join(image_dir, 'unique-seq-training-{}-{}.svg'.format(dataset, num_seq_hist)))
print("TRAINIG SET")
print("Percentage of default sequence: {:.4f}".format(seq_number[0]/len(subdf_train['case-id'].unique())))
print("Proportion of second most repeated sequence with respect to the default: {:.4f}".format(seq_number[1]/seq_number[0]))
print("Total number of traces: {}".format(len(subdf_train['case-id'].unique())))

# TOTAL DF
data_hist = []
seq_sort = df_tot.drop_duplicates('case-id').groupby('hash').count().sort_values(
    ['case-id'], ascending=False)
hash_sort = seq_sort.index.values.tolist()
seq_number = seq_sort['case-id'].values.tolist()
cases_name_sort = []
#breakpoint()
for i, unique_seq in enumerate(seq_number[:num_seq_hist]):
    name_seq = df_tot[df_tot['hash']==hash_sort[i]]['case-id'].values[0]
    if not dataset == 'Helpdesk':
        name_seq = 'Case{}'.format(name_seq)
    cases_name_sort.append(name_seq)
    data_hist.extend(unique_seq*[name_seq])
fig = go.Figure()
fig.add_trace(go.Histogram(x=data_hist))
#fig.update_layout(title='Distribution of unique sequences in {} dataset'.format(dataset))
fig.show()
#fig.update_xaxes(showticklabels=False)
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=np.arange(1, num_seq_hist+1), y=seq_number[:num_seq_hist]))
fig_bar.write_image(os.path.join(image_dir, 'unique-seq-total-df-{}-{}.svg'.format(dataset, num_seq_hist)))

print("TOTAL SET")
print("Percentage of default sequence: {:.4f}".format(seq_number[0]/len(df_tot['case-id'].unique())))
print("Proportion of second most repeated sequence with respect to the default: {:.4f}".format(seq_number[1]/seq_number[0]))
print("Total number of traces: {}".format(len(df_tot['case-id'].unique())))

data_hist = []
seq_sort = subdf_vali.drop_duplicates('case-id').groupby('hash').count().sort_values(
    ['case-id'], ascending=False)
hash_sort = seq_sort.index.values.tolist()
seq_number = seq_sort['case-id'].values.tolist()
cases_name_sort = []
for i, unique_seq in enumerate(seq_number[:num_seq_hist]):
    name_seq = subdf_vali[subdf_vali['hash']==hash_sort[i]]['case-id'].values[0]
    if not dataset == 'Helpdesk':
        name_seq = 'Case{}'.format(name_seq)
    cases_name_sort.append(name_seq)
    data_hist.extend(unique_seq*[name_seq])
fig_vali = go.Figure()
fig_vali.add_trace(go.Histogram(x=data_hist))
fig_vali.show()
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=np.arange(1, num_seq_hist+1), y=seq_number[:num_seq_hist]))
fig_bar.write_image(os.path.join(image_dir, 'unique-seq-validation-df-{}-{}.svg'.format(dataset, num_seq_hist)))

# plot sequences length
seq_len_sort = df_tot[df_tot['case-id'].isin(cases_name_sort)].pivot_table(
    index='case-id', aggfunc='size').sort_values(ascending=False)
seq_len_sort_values = seq_len_sort.values.tolist()
seq_len_sort_name = seq_len_sort.index.tolist()
data_hist = []
for i, length_seq in enumerate(seq_len_sort_values[:num_seq_hist]):
    name_seq = seq_len_sort_name[i]
    if not dataset == 'Helpdesk':
        name_seq = 'Case{}'.format(name_seq)
    data_hist.extend((length_seq+2)*[name_seq]) 
fig_len = go.Figure()
fig_len.add_trace(go.Histogram(x=data_hist))
fig_len.show()

#same thing but with validation only
# plot sequences length
seq_len_sort = subdf_vali[subdf_vali['case-id'].isin(cases_name_sort)].pivot_table(
    index='case-id', aggfunc='size').sort_values(ascending=False)
seq_len_sort_values = seq_len_sort.values.tolist()
seq_len_sort_name = seq_len_sort.index.tolist()
data_hist = []
for i, length_seq in enumerate(seq_len_sort_values):
    name_seq = seq_len_sort_name[i]
    data_hist.extend((length_seq+2)*[name_seq]) 
fig_len = go.Figure()
fig_len.add_trace(go.Histogram(x=data_hist))
fig_len.show()

#breakpoint()
most_prob_seq_case = subdf_train.sort_values(['hash-count'], ascending=False).drop_duplicates('hash')['case-id'].values
most_prob_seq_case = most_prob_seq_case[:num_most_prob_seq].tolist()
df_most_prob = subdf_train[subdf_train['case-id'].isin(most_prob_seq_case)].copy()
most_prob_seq = []
start = np.asarray([1])
#breakpoint()
for i in range(num_most_prob_seq):
    case = df_most_prob[df_most_prob['case-id'] == most_prob_seq_case[i]]['activity-role-index'].values
    seq = np.concatenate((start, case), axis=0)
    seq = np.concatenate((seq, np.array([2])), axis=0)
    seq = seq[np.newaxis, :]
    if i > 0:
        cols = np.abs(most_prob_seq.shape[1]-seq.shape[1])
        if most_prob_seq.shape[1] > seq.shape[1]:
            zeros = np.zeros((1, cols), dtype=np.int32)
            seq = np.concatenate((seq, zeros), axis=-1)
        elif most_prob_seq.shape[1] < seq.shape[1]:
            zeros = np.zeros((most_prob_seq.shape[0], cols), dtype=np.int32)
            most_prob_seq = np.concatenate((most_prob_seq, zeros), axis=-1)
        #breakpoint()
        most_prob_seq = np.concatenate((most_prob_seq, seq), axis=0)
    else:
        most_prob_seq = seq
print(most_prob_seq)
