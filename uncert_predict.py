import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import yaml
import tensorflow_datasets as tfds
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import argparse

def accuracy_function(real, pred):
    #breakpoint()
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    #breakpoint()
    return tf.reduce_sum(accuracies, axis=-1)/tf.reduce_sum(mask, axis=-1)

def single_accuracies(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    #breakpoint()
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    accuracies = tf.where(mask, accuracies, tf.ones(tf.shape(accuracies))*-1)
    return accuracies

def get_targets_probability(real, pred):
    one_hot = tf.one_hot(real, pred.shape[2])
    mult = pred * one_hot
    target_prob = tf.reshape(mult[mult>0], (real.shape[0], real.shape[1]))
    return target_prob
    

def compute_features(file_path, vocabularies):
    with open(file_path, 'r') as file:
        features = list(yaml.load_all(file, Loader=yaml.FullLoader))
    for feature in features:
        if feature['feature-type'] == 'string':
            feature['vocabulary'] = vocabularies[feature['name']]
    return features

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='choose the dataset')
parser.add_argument('model_directory', help='directory where the ensamble models are saved')
parser.add_argument('--plot_entire_sequences', help='plot the output distribution of N random sequences',
        default=0, type=int)
parser.add_argument('--plot_wrong_predictions', help='plot N output distribution of wrong predictions',
        default=0, type=int)
parser.add_argument('--dataset_type', help='choose what segment of dataset to use between training, validation and test. Default is all',
        default='all', choices=['training', 'validation', 'test', 'all'])

args = parser.parse_args()

dataset = args.dataset.lower()
model_dir = os.path.join('models_ensamble', dataset.lower(), args.model_directory)
if not os.path.isdir(model_dir):
    raise OSError('Directory {} does not exist'.format(model_dir)) 

plot_entire_seqs = False
num_seq_entire = args.plot_entire_sequences
if num_seq_entire > 0:
    plot_entire_seqs = True
plot_wrong_preds = False
num_wrong_preds = args.plot_wrong_predictions
if num_wrong_preds > 0:
    plot_wrong_preds = True
ds_type = args.dataset_type.lower()
if not dataset in ['helpdesk']:
    raise ValueError('Dataset not available') 

if dataset == 'helpdesk':
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


    features = compute_features(os.path.join(model_dir, 'features.params'), {'activity': vocabulary})
    batch_size = 64
    padded_ds_train = ds_train.padded_batch(batch_size, 
            padded_shapes=padded_shapes,
            padding_values=padding_values).prefetch(tf.data.AUTOTUNE)
    train_examples = len(ds_train)
    padded_ds_vali = ds_vali.padded_batch(batch_size, 
            padded_shapes=padded_shapes,
            padding_values=padding_values).prefetch(tf.data.AUTOTUNE)
    vali_examples = len(ds_vali)
    padded_ds_test = ds_test.padded_batch(batch_size, 
            padded_shapes=padded_shapes,
            padding_values=padding_values).prefetch(tf.data.AUTOTUNE)
    test_examples = len(ds_test)

    if ds_type == 'all':
        datasets = [(padded_ds_train, train_examples, 'training set'),
                (padded_ds_vali, vali_examples, 'validation set'),
                (padded_ds_test, test_examples, 'test set')]
    elif ds_type == 'training':
        datasets = [(padded_ds_train, train_examples, 'training set')]
    elif ds_type == 'validation':
        datasets = [(padded_ds_vali, vali_examples, 'validation set')]
    elif ds_type == 'test':
        datasets = [(padded_ds_test, test_examples, 'test set')]
    else:
        raise ValueError('Dataset type not understood')

#model_dir = 'models_ensamble/ensamble_1'

models_names = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
if len(models_names) == 0:
    raise OSError('No models are contained in {}'.format(model_dir))

models = []
for model_name in models_names:
    model_path = os.path.join(model_dir, model_name)
    model = tf.keras.models.load_model(model_path)
    models.append(model)

for ds, num_examples, ds_name in datasets:
    if plot_entire_seqs or plot_wrong_preds:
        count_seq = 0
        count_wrong = 0
        random_idx_plot = random.sample(range(num_examples), num_seq_entire)
    u_t_array_mean = np.zeros(1)
    u_t_array_single = np.zeros(1)
    u_a_array_mean = np.zeros(1)
    u_a_array_single = np.zeros(1)
    u_e_array_mean = np.zeros(1)
    u_e_array_single = np.zeros(1)
    acc_array_mean = np.zeros(1)
    acc_array_single = np.zeros(1)
    target_prob_array = np.zeros(1)
    for batch_idx, batch_data in enumerate(ds):
        #breakpoint()
        #if batch_data['variant'][0, 0] == 92 or COMPUTE_ALL:
        input_data = []
        for feature in features:
            input_data.append(batch_data[feature['name']][:, :-1])
        target_data = batch_data['activity'][:, 1:]
        target_data = output_preprocess(target_data)
        mask = tf.math.logical_not(tf.math.equal(target_data, 0))

        u_a = 0
        for i, model in enumerate(models):
            out_prob = tf.nn.softmax(model(input_data))
            if  i == 0:
                out_prob_tot_distr = out_prob
            else:
                out_prob_tot_distr += out_prob
            # compute aleatoric uncertainty
            u_a += np.sum(out_prob.numpy()*np.log(out_prob.numpy()), axis=-1)

        out_prob_tot_distr /= len(models)

        #breakpoint()
        acc = accuracy_function(target_data, out_prob_tot_distr)
        acc_array_mean = np.hstack([acc_array_mean, acc])

        acc_single = single_accuracies(target_data, out_prob_tot_distr)
        acc_array_single = np.hstack([acc_array_single, acc_single.numpy().ravel()])
        
        target_prob = get_targets_probability(target_data, out_prob_tot_distr)
        target_prob_array = np.hstack([target_prob_array, target_prob.numpy().ravel()])
        #breakpoint()

        u_a /= -len(models)
        u_a = np.where(mask, u_a, np.zeros(np.shape(u_a)))
        u_a_array_single = np.hstack([u_a_array_single, u_a.ravel()])
        # compute total uncertainty
        u_t = -np.sum(out_prob_tot_distr.numpy() * np.log(out_prob_tot_distr.numpy()), axis=-1)
        u_t = np.where(mask, u_t, np.zeros(np.shape(u_t)))
        u_t_array_single = np.hstack([u_t_array_single, u_t.ravel()])
        # compute epistemic uncertainty
        u_e = u_t - u_a
        u_e_array_single = np.hstack([u_e_array_single, u_e.ravel()])
        #breakpoint()
#out_prob_tot_distr = tf.nn.softmax(out_prob_tot)

        vocabulary_plot = ['<PAD>', '<START>',  '<END>','Resolve SW anomaly', 'Resolve ticket', 'RESOLVED', 
                      'DUPLICATE', 'Take in charge ticket', 'Create SW anomaly',
                      'Schedule intervention', 'VERIFIED', 'Closed', 'Wait',
                      'Require upgrade', 'Assign seriousness', 'Insert ticket', 'INVALID']
#        for i in range(5):
#            batch_seq = out_prob_tot_distr[i, :]
        #breakpoint()
        length_seq = tf.reduce_sum(
                tf.cast(tf.math.logical_not(
                    tf.math.equal(target_data, 0)), dtype=tf.float32),
                axis=-1)
        mean_u_t = np.sum(u_t, axis=-1) / length_seq.numpy()
        mean_u_a = np.sum(u_a, axis=-1) / length_seq.numpy()
        mean_u_e = np.sum(u_e, axis=-1) / length_seq.numpy()
        #breakpoint()
        u_t_array_mean = np.hstack([u_t_array_mean, mean_u_t])
        u_a_array_mean = np.hstack([u_a_array_mean, mean_u_a])
        u_e_array_mean = np.hstack([u_e_array_mean, mean_u_e])
        check_cond_wrong = np.logical_and(acc_single==0.0, mask).any() and plot_wrong_preds and count_wrong<num_wrong_preds
        check_cond_entire = plot_entire_seqs and count_seq<num_seq_entire
        #breakpoint()
        if check_cond_wrong or check_cond_entire:
            for num_row in range(acc_single.shape[0]):
                check_cond_wrong_row = np.logical_and(acc_single[num_row, :]==0.0, mask[num_row, :]).any() and check_cond_wrong
                check_cond_random_idx = (batch_idx * batch_size + num_row in random_idx_plot) 
                if check_cond_wrong_row or check_cond_random_idx:
                    act = ''
                    for j in range(target_data.shape[1]):
                        act += '{} - '.format(input_data[0][num_row, j].numpy().decode("utf-8"))
                        check_cond_wrong_single_pred = check_cond_wrong_row and acc_single[num_row, j] == 0.0 and count_wrong<num_wrong_preds
                        if (check_cond_wrong_single_pred or check_cond_random_idx) and not target_data[num_row, j].numpy() == 0 :
                            target_numpy = np.zeros(len(vocabulary_plot))
                            target_numpy[target_data[num_row, j].numpy()] = 1
                            #fig = go.Figure()
                            fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2])
                            fig.add_trace(go.Bar(x=vocabulary_plot, y=target_numpy,
                                                 marker=dict(opacity=0.4), name='actual event'),
                                          row=1, col=1)
                            fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob[num_row, j].numpy(),
                                                 marker=dict(opacity=0.4), name='single model'),
                                          row=1, col=1)
                            fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob_tot_distr[num_row, j].numpy(),
                                                 marker=dict(opacity=0.4), name='ensamble models'),
                                          row=1, col=1)
                            fig.add_trace(go.Bar(x=['Uncertainty'], y=[u_t[num_row, j]], name='Epistemic', offsetgroup=0),
                                          row=1, col=2)
                            fig.add_trace(go.Bar(x=['Uncertainty'], y=[u_a[num_row, j]], name='Aleatoric', offsetgroup=0),
                                          row=1, col=2)
                            fig.layout['yaxis2'].update(range=[0, 1.2*np.max(u_t)])
                            fig.update_layout(barmode='overlay', title_text="{}<br><sup>{}</sup>".format(ds_name.capitalize(), act))
                            #fig.update_traces(opacity=0.6)
                            fig.show(renderer='chromium')
                            #breakpoint()
                            count_wrong += 1
                    count_seq += 1
#                    print("Mean total uncertainty: {}".format(mean_u_t[i]))
#                    print("Mean aleatoric uncertainty: {}".format(mean_u_a[i]))
#                    print("Mean epistemic uncertainty: {}".format(mean_u_e[i]))
                    #breakpoint()

    u_t_array = u_t_array_mean[1:]
    u_a_array = u_a_array_mean[1:]
    u_e_array = u_e_array_mean[1:]
    acc_array = acc_array_mean[1:]
    acc_array_single = acc_array_single[1:]
    u_t_array_single = u_t_array_single[1:]
    u_a_array_single = u_a_array_single[1:]
    u_e_array_single = u_e_array_single[1:]
    target_prob_array = target_prob_array[1:]
    #breakpoint()
# ordering
    ordered_acc_array = acc_array_single.argsort()
    acc_array_single = acc_array_single[ordered_acc_array]
    u_t_array_single = u_t_array_single[ordered_acc_array]
    u_a_array_single = u_a_array_single[ordered_acc_array]
    u_e_array_single = u_e_array_single[ordered_acc_array]
    target_prob_array = target_prob_array[ordered_acc_array]

    group_labels = ['Total uncertainty', 'Aleatoric uncertainty', 'Epistemic uncertainty']
    hist_data = [u_t_array, u_a_array, u_e_array]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.02)
    fig.update_layout(title_text=ds_name.capitalize())
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_t_array, y=acc_array, mode='markers', name='Total uncertainty'))
    fig.update_layout(title_text=ds_name.capitalize())
    fig.update_xaxes(title_text='Total uncertainty')
    fig.update_yaxes(title_text='Mean sequence accuracy')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_a_array, y=acc_array, mode='markers', name='Aleatoric uncertainty'))
    fig.update_layout(title_text=ds_name.capitalize())
    fig.update_yaxes(title_text='Mean sequence accuracy')
    fig.update_xaxes(title_text='Aleatoric uncertainty')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_e_array, y=acc_array, mode='markers', name='Epistemic uncertainty'))
    fig.update_layout(title_text=ds_name.capitalize())
    fig.update_yaxes(title_text='Mean sequence accuracy')
    fig.update_xaxes(title_text='Epistemic uncertainty')
    fig.show(renderer='chromium')

# select valid unceratinties and divide between right and wrong predictions
    #breakpoint()
    u_t_array_single = u_t_array_single[acc_array_single>-1]
    u_a_array_single = u_a_array_single[acc_array_single>-1]
    u_e_array_single = u_e_array_single[acc_array_single>-1]
    target_prob_array = target_prob_array[acc_array_single>-1]
    acc_array_single = acc_array_single[acc_array_single>-1]
    u_t_array_single_right = u_t_array_single[acc_array_single == 1]
    u_t_array_single_wrong = u_t_array_single[acc_array_single == 0]
    u_a_array_single_right = u_a_array_single[acc_array_single == 1]
    u_a_array_single_wrong = u_a_array_single[acc_array_single == 0]
    u_e_array_single_right = u_e_array_single[acc_array_single == 1]
    u_e_array_single_wrong = u_e_array_single[acc_array_single == 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_t_array_single[acc_array_single>-1], y=target_prob_array,
        marker=dict(color=acc_array_single, colorbar=dict(title='Point prediction correctness'), colorscale='Viridis'),
        mode='markers'))
    fig.update_layout(title_text=ds_name.capitalize())
    fig.update_xaxes(title_text='Total uncertainty')
    fig.update_yaxes(title_text='Assigned probability')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_t_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
        marker=dict(color=target_prob_array, colorbar=dict(title='Assinged probability'), colorscale='Viridis'),
        mode='markers'))
    fig.update_layout(title_text=ds_name.capitalize())
    fig.update_xaxes(title_text='Total uncertainty')
    fig.update_yaxes(title_text='Point prediction correctness')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_a_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
        marker=dict(color=target_prob_array, colorbar=dict(title='Assinged probability'), colorscale='Viridis'),
        mode='markers'))
    fig.update_layout(title_text=ds_name.capitalize())
    fig.update_xaxes(title_text='Aleatoric uncertainty')
    fig.update_yaxes(title_text='Point prediction correctness')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_e_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
        marker=dict(color=target_prob_array, colorbar=dict(title='Assinged probability'), colorscale='Viridis'),
        mode='markers'))
    fig.update_layout(title_text=ds_name.capitalize())
    fig.update_xaxes(title_text='Epistemic uncertainty')
    fig.update_yaxes(title_text='Point prediction correctness')
    fig.show(renderer='chromium')

# Plot distribution
    group_labels = ['Right predictions', 'Wrong predictions']
    hist_data = [u_t_array_single_right, u_t_array_single_wrong]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.002)
    fig.update_layout(title_text="{} - Total uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')

    hist_data = [u_a_array_single_right, u_a_array_single_wrong]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.002)
    fig.update_layout(title_text="{} - Aleatoric uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')

    hist_data = [u_e_array_single_right, u_e_array_single_wrong]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.002)
    fig.update_layout(title_text="{} - Epistemic uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')

# Boxplot
    fig = go.Figure()
    fig.add_trace(go.Box(y=u_t_array_single_right, name='Right predictions'))
    fig.add_trace(go.Box(y=u_t_array_single_wrong, name='Wrong predictions'))
    fig.update_layout(title_text="{} - Total uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_a_array_single_right, name='Right predictions'))
    fig.add_trace(go.Box(y=u_a_array_single_wrong, name='Wrong predictions'))
    fig.update_layout(title_text="{} - Aleatoric uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_e_array_single_right, name='Right predictions'))
    fig.add_trace(go.Box(y=u_e_array_single_wrong, name='Wrong predictions'))
    fig.update_layout(title_text="{} - Epistemic uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_t_array_single))
    fig.update_layout(title_text="{} - Total uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_a_array_single))
    fig.update_layout(title_text="{} - Aleatoric uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_e_array_single))
    fig.update_layout(title_text="{} - Epistemic uncertainty".format(ds_name.capitalize()))
    fig.show(renderer='chromium')
