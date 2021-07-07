import pickle
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from preprocessing_utils import EventLogImporter, RolesDiscover, CreateSequences
from model_utils import create_mask, positional_encodings
from model_types import Transformer
import os
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_hash(x):
    return hash(tuple(x))

save_results_for_app = False
# interest case validation 'Case4567'
dataset = 'Helpdesk'
#dataset = 'BPI-Challenge-2012'
data_dir = './data'
data_app_dir = './app_data'
model_dir = './models'
image_dir = './images'
#HELPDESK
case = 'Case114'
case = 'Case49'
case = 'Case135'
#case = 'Case4567'
#case = 'Case1528'
#case = 'Case2975'
#case = 'Case17'
#case = 'Case82'
#case = 'Case114'
#case = 'Case2'

#BPI
#case = 'Case173697'
#case = 'Case173763'
#case = 'Case173853'
if not dataset == 'Helpdesk':
    case = case.split('Case')[1]
res_data = pd.read_pickle('results/results-df.pkl')
#mod_data = res_data.iloc[91]
mod_data = res_data.iloc[117]
path_df = os.path.join(data_dir, '{}.xes'.format(dataset).replace('-', '_'))
hash_code = mod_data['hash']
path_model = os.path.join(model_dir, dataset.lower(), 'type{}'.format(str(mod_data['embedding_type'])),
                          'predictive-model', hash_code, 'fold1')
#embedding_path = os.path.join(model_dir, dataset.lower(), 'type3', 'embedding', 'activity-role-embedding.npy')
embedding_path = None
index_train_path = os.path.join(model_dir, dataset.lower(), 'type{}'.format(str(mod_data['embedding_type'])), 
                                'predictive-model', hash_code,  'fold1', 'train_indeces.pkl') 
index_val_path = os.path.join(model_dir, dataset.lower(), 'type{}'.format(str(mod_data['embedding_type'])), 
                              'predictive-model', hash_code, 'fold1', 'val_indeces.pkl') 
colors = ['red', 'blue', 'grey', 'green', 'brown',
          'cadetblue', 'navy', 'cornflowerblue', 'darkslategrey', 'teal']

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

vocabulary = df.iloc[df['activity-role-index'].drop_duplicates().index][[
    'task', 'role', 'activity-role-index']]
vocabulary = vocabulary.append({'task': '<START>', 'role': 0, 'activity-role-index': 1}, ignore_index=True)
vocabulary = vocabulary.append({'task': '<END>', 'role': 0, 'activity-role-index': 2}, ignore_index=True)
vocabulary = vocabulary.append({'task': '<PAD>', 'role': 0, 'activity-role-index': 0}, ignore_index=True)
x_label =[]
for i in range(len(vocabulary)):
    name_event = vocabulary.loc[vocabulary['activity-role-index']==i]
    x_label.append("{}-role{}".format(name_event['task'].values[0], name_event['role'].values[0]))

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

sequence = [1]
#sequence.extend(df_vali[df_vali['case-id'] == 'Case4567']['activity-role-index'])
sequence.extend(df_vali[df_vali['case-id'] == case]['activity-role-index'])
sequence.extend([2])
#breakpoint()
# create the plot
# plot sequence of rectangles
if save_results_for_app:
    length_input_seq = len(sequence)-1
else:
    length_input_seq = 2

attention_weights_total = {}
df_app = pd.DataFrame(columns=['head', 'act_role','layer','num_sample', 'att_weight', 'num_step'])
df_pred = pd.DataFrame(columns=['num_step', 'actual_value', 'pred_value', 'percentage', 'num_sample'])
for z in range(1, len(sequence)):
    print("Step {} of {}".format(z, len(sequence)))
    length_input_seq = z
    length_rect = 1
    dist_rect = 0.1
    CHOOSE_LAYER = 1
    NUM_SAMPLES = 10
    attention_weights_total['step{}'.format(z)] = {}
    if length_input_seq < 4:
        font_size = 16
    else:
        font_size = 14
    x0 = 0
    y0 = 0
    x1 = -length_rect
    y1 = 1
    x0_arc =  dist_rect
    y0_arc = 1
    cont = 2
#breakpoint()
    activities = ['<START>']
    activities.extend(df_vali[df_vali['case-id'] == case]['task'])
    activities.extend(['<END>'])
    roles = [0]
    roles.extend(df_vali[df_vali['case-id'] == case]['role'].values.tolist())
    roles.extend([0])
    roles = list(map(lambda x: 'role {}'.format(x), roles))
    input_act = activities[0:length_input_seq]
    input_rol = roles[0:length_input_seq]
    input_seq = sequence[0:length_input_seq]
# initialize
    prediction_entropy = []
    att_weights_total = {}
    for i in range(NUM_LAYERS):
        key = 'layer{}'.format(str(i+1))
        att_weights_total[key] = {}
        for j in range(num_heads):
            att_weights_total[key]['head{}'.format(str(j+1))] = np.zeros((NUM_SAMPLES, len(input_seq)))
    #fig_bars = go.Figure()
    fig_unc = make_subplots(rows=1, cols=2)
    for k in range(NUM_SAMPLES):
        print(k)
# make real predictions 
        input_np = np.asarray(input_seq)
        input_np = input_np[np.newaxis, :]
        input_tf = tf.convert_to_tensor(input_np)
        mask = create_mask(input_tf)
        predictions, attention_weights = predictive_model(input_tf, True, mask)
        layer_att = attention_weights['decoder_layer{}_block'.format(CHOOSE_LAYER)].numpy()
        pred_softmax = tf.nn.softmax(predictions).numpy()[0,-1,:]
        prediction_entropy.append(- np.sum(pred_softmax*np.log(pred_softmax)))
        for el in range(pred_softmax.shape[0]):
            record = {'num_step': z, 'actual_value': sequence[z], 'pred_value': el, 
                      'percentage': pred_softmax[el], 'num_sample': k}
            df_pred = df_pred.append(record, ignore_index=True)

        # output bars
        colors_pred = len(vocabulary)*['blue']
        colors_pred[sequence[z]] = 'red'
        #breakpoint()
#        fig_bars.add_trace(go.Bar(x=np.arange(len(vocabulary)), y=pred_softmax, marker_color=colors_pred,
#                                  opacity=0.1, showlegend=False))
#        fig_unc.add_trace(go.Bar(x=np.arange(len(vocabulary)), y=pred_softmax, marker_color=colors_pred,
#                                  opacity=0.1, showlegend=False), row=1, col=1)
        fig_unc.add_trace(go.Bar(x=x_label, y=pred_softmax, marker_color=colors_pred,
                                  opacity=0.1, showlegend=False), row=1, col=1)

        for i in range(NUM_LAYERS):
            for j in range(num_heads):
                att_weights_total['layer{}'.format(str(i+1))]['head{}'.format(str(j+1))][k, :] = \
                    attention_weights['decoder_layer{}_block'.format(str(i+1))][0, j, -1].numpy()
                # cols = len input seq
                array_weights = attention_weights['decoder_layer{}_block'.format(str(i+1))][0, j, -1].numpy()
                steps = array_weights.shape[0]
                for step in range(steps):
                    rep = 0
                    for check_step in range(step):
                        if input_act[check_step] == input_act[step] and input_rol[check_step] == input_rol[step]:
                            rep += 1 
                    if rep > 0:
                        name_step = '{} - {} ({})'.format(input_act[step], input_rol[step], rep)
                    else: 
                        name_step = '{} - {}'.format(input_act[step], input_rol[step])
                    record = {'head': j+1, 'act_role': name_step, 'layer': i+1, 
                              'num_sample': k+1, 'att_weight': array_weights[step], 
                              'num_step': z}
                    df_app = df_app.append(record, ignore_index=True)


        attention_weights_total['step{}'.format(z)]['sample{}'.format(k)] = attention_weights

#    fig_bars.update_layout(barmode='overlay')
#    fig_bars.update_yaxes(range=[-0.1, 1.1])
#    fig_bars.show(renderer='chromium')
#    fig_bars.write_image(os.path.join(image_dir, 'predict-default-seq-{}-step-{}.svg'.format(dataset, z)))
#
#    fig_en = go.Figure()
#    fig_en.add_trace(go.Box(y=prediction_entropy))#, boxmean='sd'))
#    fig_en.update_yaxes(range=[0, 2])
#    fig_en.show(renderer='chromium')
#    fig_en.write_image(os.path.join(image_dir, 'entropy-predict-default-seq-{}-step-{}.svg'.format(dataset, z)))

    fig_unc.add_trace(go.Box(y=prediction_entropy, showlegend=False, boxmean='sd'), row=1, col=2)#))
    fig_unc.update_layout(barmode='overlay', yaxis=dict(range=[-0.1, 1.1]), 
                          yaxis2=dict(range=[0, 2]), xaxis2=dict(showticklabels=False))
    fig_unc.write_image(os.path.join(image_dir, 'pred-distr-{}-{}-step{}.svg'.format(dataset, case, z)))

    activities_role_name = []
    for i in range(len(input_seq)):
        # if the event is a repetition add number_of_repetition
        rep = 0
        for j in range(i):
            if input_act[j] == input_act[i] and input_rol[j] == input_rol[i]:
                rep += 1 
        if rep > 0:
            activities_role_name.extend(NUM_SAMPLES*['{} - {} ({})'.format(
                input_act[i], input_rol[i], rep)])
        else:
            activities_role_name.extend(NUM_SAMPLES*['{} - {}'.format(
                input_act[i], input_rol[i])])

    fig_att = make_subplots(rows=NUM_LAYERS, cols=1)
    for i in range(NUM_LAYERS):
        if i == 0:
            legend = True
        else:
            legend = False
        for j in range(num_heads):
            data = att_weights_total['layer{}'.format(str(i+1))][
                'head{}'.format(str(j+1))].transpose().reshape((NUM_SAMPLES*len(input_seq), ))
            fig_att.add_trace(go.Box(x=activities_role_name, y=data, 
                                     name='head{}'.format(str(j+1)),
                                     marker_color=colors[j], showlegend=legend,
                                     offsetgroup=j, boxmean='sd'),
                              row=i+1, col=1)
    fig_att.update_layout(boxmode='group')
    fig_att.update_yaxes(range=[-0.1, 1.1])
    fig_att.show(renderer='chromium')
    fig_att.write_image(os.path.join(image_dir, 'att-weight-{}-case{}-step-{}.svg'.format(dataset, case, z)))

    fig_unc.show(renderer='chromium')

if save_results_for_app:
    file_name = os.path.join(data_app_dir, 'attention_weights_{}.pickle'.format(case))
    with open(file_name, 'wb') as handle:
        pickle.dump(attention_weights_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = os.path.join(data_app_dir, 'activities_{}.pickle'.format(case))
    with open(file_name, 'wb') as handle:
        pickle.dump(activities, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = os.path.join(data_app_dir, 'roles_{}.pickle'.format(case))
    with open(file_name, 'wb') as handle:
        pickle.dump(roles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = os.path.join(data_app_dir, 'vocabulary_app.pickle')
    vocabulary.to_pickle(file_name)
    file_name = os.path.join(data_app_dir, 'df_app.pickle')
    df_app.to_pickle(file_name)
    file_name = os.path.join(data_app_dir, 'df_pred.pickle')
    df_pred.to_pickle(file_name)
