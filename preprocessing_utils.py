import pm4py
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import random

def transform_log_normal(data):
    log_data = np.log1p(data)
    max_value = np.max(log_data)
    min_value = np.min(log_data)
    log_data_norm = np.divide(log_data, max_value) if max_value > 0 else 0
    return log_data_norm, {'max value': max_value, 'min value': min_value}

def transform_max(data):
    max_value = np.max(data)
    data_norm = np.divide(data, max_value) if max_value > 0 else 0
    return data_norm, {'max value': max_value}

def split_data(df, perc_train, perc_vali, perc_test):
    if perc_train + perc_vali + perc_test != 1.:
        raise ValueError('Percentage not correct')
    else:
        total_cases = shuffle(list(df['case-id'].unique()))
        num_train = int(np.round(perc_train*len(total_cases)))
        num_vali = int(np.round(perc_vali*len(total_cases)))
        num_test = int(np.round(perc_test*len(total_cases)))
        case_train = total_cases[:num_train]
        case_vali = total_cases[num_train:num_train+num_vali]
        case_test = total_cases[num_train+num_vali:]
    df_train = df[df['case-id'].isin(case_train)]
    df_vali = df[df['case-id'].isin(case_vali)]
    df_test = df[df['case-id'].isin(case_test)]
    return df_train, df_vali, df_test


class EventLogImporter:
    def __init__(self, log_path, norm_method):
        if 'xes' in log_path:
            event_log = pm4py.read_xes(log_path)
        elif 'csv' in log_path:
            event_log = pm4py.read_csv(logPath)
        else:
            raise TypeError(log_path)
        self._df = pm4py.convert_to_dataframe(event_log)
        self._check_nan()
        self._standardize_df()
#        self._compute_relative_time()
#        self._normalize_relative_time(norm_method)
        self._compute_act_index()
        #self._compute_act_role_index()

    def _check_nan(self):
        self._df['org:resource'].fillna('auto', inplace=True)

    def _standardize_df(self):
        # Transforms the _df in a _df with columns ['case id', 'task', 'user', 'relative time'] 
        if len(self._df['lifecycle:transition'].unique()) > 1:
            self._df['task'] = self._df['concept:name'] + '_' + self._df['lifecycle:transition']
        else:
            self._df['task'] = self._df['concept:name']
        self._df['case-id'] = self._df['case:concept:name']
        self._df['user'] = self._df['org:resource']

    def _compute_relative_time(self):
        self._df['relative-time'] = self._df['time:timestamp'].diff()
        cases = self._df['case-id'].unique()
        for case in tqdm(cases):
            index = self._df[self._df['case-id'] == case].index[0]
            self._df.loc[index, 'relative-time'] = pd.Timedelta(0)

    def _normalize_relative_time(self, norm_method):
        if norm_method == 'log-normal':
            self._df['relative-time-norm'], self.norm_params = transform_log_normal(
                self._df['relative-time'].dt.seconds)
        elif norm_method == 'max':
            self._df['relative-time-norm'], self.norm_params = transform_max(
                self._df['relative-time'].dt.seconds)
        else:
            raise ValueError(norm_method)
    
    def _compute_act_index(self):
        for i, activity in enumerate(self._df['task'].unique()):
            #mask = self._df['task'] == activity
            index = self._df[self._df['task'] == activity].index
            # adding 3 because 0,1,2 are reserved to <START> <END> <PAD>
            self._df.loc[index, 'activity-index'] = i + 3
        self._df = self._df.astype({'activity-index': 'int32'})

#    def get_df(self, columns=['case-id', 'task', 'user', 'relative-time', 
#                             'relative-time-norm', 'activity-index']):
#        return self._df[columns].copy()
    def get_df(self, columns=['case-id', 'task', 'user', 'activity-index']):
        return self._df[columns].copy()


class RolesDiscover:
    def __init__(self, df, threshold=0.7):
        # df in 'standard' form: ['case id', 'task', 'user', 'relative time']
        self._df = df
        self._compute_act_matrix()
        self.threshold = threshold
        self.discover()
        self.add_roles()

    def _compute_act_matrix(self):
        freq_array = []
        self._act_matrix = pd.DataFrame()
        for user in self._df['user'].unique():
            freq_array = []
            for task in self._df['task'].unique():
                freq_array.append(len(self._df[
                    (self._df['user'] == user) & (self._df['task'] == task)]))                
            self._act_matrix[user] = freq_array
        self._corr_matrix = self._act_matrix.corr()

    def discover(self):
        g = nx.Graph()        
        for user in self._df['user'].unique():
            g.add_node(user)
        for first_user in self._corr_matrix.keys():
            for second_user in self._corr_matrix.keys():
                coeff = self._corr_matrix[first_user][second_user]
                if not first_user == second_user and coeff > self.threshold:
                    g.add_edge(first_user, second_user, weight= coeff)
        self._sub_graphs = list(nx.connected_components(g))

    def add_roles(self):
        for i, sub_graph in enumerate(self._sub_graphs):
            for user in sub_graph:
                index = self._df[self._df['user'] == user].index
                # adding 1 because the group 0 is reserved
                self._df.loc[index, 'role'] = i + 1
                #mask = self._df['user'] == user
                #self._df.loc[mask, 'role'] = i
        self._df = self._df.astype({'role': 'int32'})
        self._compute_act_role_index()

    def _compute_act_role_index(self):
        for i, index in enumerate(self._df[['task', 'role']].drop_duplicates().index):
            # adding 3 because 0, 1, 2, are reserved to <PAD> <START> <END>
            total_index = self._df[(self._df['task'] == self._df.loc[index, 'task'])
                                    & (self._df['role'] == self._df.loc[index, 'role'])].index
            self._df.loc[total_index, 'activity-role-index'] = i + 3
        self._df = self._df.astype({'activity-role-index': 'int32'})
    
    def get_df(self):
        return self._df.copy()

    def get_corr_matrix(self):
        return self._corr_matrix.copy()

    def get_sub_graphs(self): 
        return self._sub_graphs


class CreateSequences:
    def __init__(self, df):
        self._df = df

    def _extract_seq(self, group_df, case):
        seq = group_df.get_group(case)['activity-role-index'].values
        case_len = seq.shape[0]
        case_arr = np.ones((case_len+2,), dtype=int)
        case_arr[1:-1] = seq
        case_arr[-1] = 2
        return case_arr

    def generate_data(self, batch_size):
        total_cases = self._df['case-id'].unique()
        while True:
            cases = np.random.choice(total_cases, size=batch_size)
            sub_df = self._df[self._df['case-id'].isin(cases)].copy()
            max_len = max(sub_df.groupby(['case-id']).count()['activity-role-index'].values)
            batch = np.zeros((batch_size, max_len+2), dtype=int)
            group_df = sub_df.groupby(['case-id'])
            for i, case in enumerate(cases):
                seq = self._extract_seq(group_df, case)
                batch[i, 0:seq.shape[0]] = seq
            yield batch
