import tensorflow as tf
import yaml
import tensorflow_datasets as tfds
import os
import plotly.graph_objects as go
import numpy as np

ds_train = tfds.load('helpdesk', split='train[:70%]', shuffle_files=True)
ds_vali = tfds.load('helpdesk', split='train[70%:85%]')
ds_test = tfds.load('helpdesk', split='train[85%:]')

vocabulary = ['<START>',  '<END>','Resolve SW anomaly', 'Resolve ticket', 'RESOLVED', 
              'DUPLICATE', 'Take in charge ticket', 'Create SW anomaly',
              'Schedule intervention', 'VERIFIED', 'Closed', 'Wait',
              'Require upgrade', 'Assign seriousness', 'Insert ticket', 'INVALID']

output_preprocess = tf.keras.layers.StringLookup(vocabulary=vocabulary,
                                                 num_oov_indices=1)

padded_shapes = {
    'activity': [None],
    'resource': [None],
    'product': [None],    
    'customer': [None],    
    'responsible_section': [None],    
    'service_level': [None],    
    'service_type': [None],    
    'seriousness': [None],    
    'workgroup': [None],
    'variant': [None],    
    'relative_time': [None],
    'day_part': [None],
    'week_day': [None]
}
padding_values = {
    'activity': '<PAD>',
    'resource': tf.cast(0, dtype=tf.int64),
    'product': tf.cast(0, dtype=tf.int64),    
    'customer': tf.cast(0, dtype=tf.int64),    
    'responsible_section': tf.cast(0, dtype=tf.int64),    
    'service_level': tf.cast(0, dtype=tf.int64),    
    'service_type': tf.cast(0, dtype=tf.int64),    
    'seriousness': tf.cast(0, dtype=tf.int64),    
    'workgroup': tf.cast(0, dtype=tf.int64),
    'variant': tf.cast(0, dtype=tf.int64),    
    'relative_time': tf.cast(0, dtype=tf.int32),
    'day_part': tf.cast(0, dtype=tf.int64),
    'week_day': tf.cast(0, dtype=tf.int64),
}

def compute_features(file_path, vocabularies):
    with open(file_path, 'r') as file:
        features = list(yaml.load_all(file, Loader=yaml.FullLoader))
    for feature in features:
        if feature['feature-type'] == 'string':
            feature['vocabulary'] = vocabularies[feature['name']]
    return features

features = compute_features('act_res_var_time.params', {'activity': vocabulary})
batch_size = 16
padded_ds = ds_train.padded_batch(batch_size, 
        padded_shapes=padded_shapes,
        padding_values=padding_values).prefetch(tf.data.AUTOTUNE)
padded_ds_vali = ds_vali.padded_batch(batch_size, 
                                      padded_shapes=padded_shapes,
                                      padding_values=padding_values).prefetch(tf.data.AUTOTUNE)

model_dir = 'models_ensamble/ensamble_1'
models_names = os.listdir(model_dir)
models = []
for model_name in models_names:
    model_path = os.path.join(model_dir, model_name)
    model = tf.keras.models.load_model(model_path)
    models.append(model)

#for step, batch_data in enumerate(tqdm(padded_ds_vali, desc='Vali', position=1, leave=False)):
batch_data = padded_ds_vali.take(1).get_single_element()
input_data = []
for feature in features:
    input_data.append(batch_data[feature['name']][:, :-1])
target_data = batch_data['activity'][:, 1:]
target_data = output_preprocess(target_data)

for i, model in enumerate(models):
    out_prob = model(input_data)
    if  i == 0:
        out_prob_tot = out_prob
        out_prob_distr = tf.nn.softmax(out_prob)
    else:
        out_prob_tot_distr += tf.nn.softmax(out_prob)

out_prob_tot_distr /= len(models)
#out_prob_tot_distr = tf.nn.softmax(out_prob_tot)

vocabulary_plot = ['<PAD>', '<START>',  '<END>','Resolve SW anomaly', 'Resolve ticket', 'RESOLVED', 
              'DUPLICATE', 'Take in charge ticket', 'Create SW anomaly',
              'Schedule intervention', 'VERIFIED', 'Closed', 'Wait',
              'Require upgrade', 'Assign seriousness', 'Insert ticket', 'INVALID']
for i in range(5):
    batch_seq = out_prob_tot_distr[i, :]
    act = ''
    for j in range(target_data.shape[1]):
        if not target_data[i, j].numpy() == 0:
            act += '{} - '.format(input_data[0][i, j].numpy().decode("utf-8"))
            target_numpy = np.zeros(len(vocabulary_plot))
            target_numpy[target_data[i, j].numpy()] = 1
            fig = go.Figure()
            fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob_distr[i, j].numpy(), name='single model'))
            fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob_tot_distr[i, j].numpy(), name='ensamble models'))
            fig.add_trace(go.Bar(x=vocabulary_plot, y=target_numpy, name='actual event'))
            fig.update_layout(barmode='overlay', title_text=act)
            fig.update_traces(opacity=0.6)
            fig.show(renderer='chromium')
