import tensorflow as tf
import datetime
import os 
import pandas as pd
import random
import time
import numpy as np
from model_types import Transformer
from preprocessing_utils import EventLogImporter, RolesDiscover, CreateSequences
import math
from training_utils import CustomSchedule, loss_function, accuracy_function
from model_utils import create_mask, positional_encodings
from fastDamerauLevenshtein import damerauLevenshtein as DL
from tqdm import tqdm
import pickle

# constant variables
EPOCHS = 2000
NUM_FOLDS = 1
DATA_DIR = './data'
MODELS_DIR = './models'
RESULTS_DIR = './results'
CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'checkpoints-train')
NORMALIZATION = 'log-normal'
NUM_LAYERS = 4
NUM_UNIT_DFF = 256
PE = 1000
BATCH_SIZE = 64
PERC_TRAIN = 0.8
PERC_VAL = 0.2
MAX_PATIENCE = 200
emb_dim_mul = 4
#DATASETS= ['Helpdesk.xes', 'BPI_2012_W_complete.xes', 'BPI_Challenge_2012.xes']
DATASETS= ['BPI_Challenge_2012.xes']
#DATASETS= ['Helpdesk.xes']
#EMBEDDING_TYPES = [1, 2, 3]
EMBEDDING_TYPES = [4]
TRAINABLE_EMB = True

df_dir = os.path.join(RESULTS_DIR, 'results-df.pkl')
if os.path.exists(df_dir):
    results_df = pd.read_pickle(df_dir)
else:
    results_df = pd.DataFrame(columns=['hash', 'fold', 'embedding_type', 'generation_type', 'units', 
                                       'layers', 'batch_size', 'dl_activities', 'dl_roles', 
                                       'train_loss', 'val_loss', 'train_accuracy', 
                                       'val_accuracy', 'epochs_done', 'max_epochs', 
                                       'patience',  'max_patience', 'dataset'])
# training all models
for dataset in DATASETS:
    # create df
    print("*** Importing {} ***".format(dataset))
    path_df = os.path.join(DATA_DIR, dataset)
    event_log = EventLogImporter(path_df, NORMALIZATION)
    df = event_log.get_df()
    role_discover = RolesDiscover(df)
    df = role_discover.get_df()

    target_vocab_size = len(df['activity-role-index'].unique()) + 3
    d_model = math.ceil(target_vocab_size**0.25)*emb_dim_mul
    steps_per_epoch = int(np.round(len(df['case-id'].unique()) / BATCH_SIZE))
        
    # number of heads is the maximum divisor less then 10 (rule decided by me)
    cont = 10
    found = False
    while not found:
        mod = d_model % cont
        if mod == 0:
            found = True
            num_heads = cont
        cont -= 1
    #num_heads = 10
    for embedding_type in EMBEDDING_TYPES:
        # name of the model is the hash of the training datetime
        hash_code = hash(datetime.datetime.now())
        print("---> Dataset: {} - Embedding type {}".format(dataset, embedding_type))
        dataset_dir_name = dataset.split('.')[0].lower().replace('_', '-')
        model_path = os.path.join(MODELS_DIR, dataset_dir_name, 
                'type{}'.format(embedding_type), 'predictive-model', str(hash_code))
        #embedding_path = os.path.join(MODELS_DIR, dataset_dir_name, 'type{}'.format(embedding_type), 'embedding', 'activity-role-embedding.npy')
        if embedding_type == 4:
            embedding_path = None
        else:
            embedding_path = os.path.join(MODELS_DIR, dataset_dir_name, 'type{}'.format(embedding_type), 'embedding', 
                    'activity-role-embedding-{}dim.npy'.format(d_model))

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # define model (PE=max positional encoding)
        if embedding_type == 4:
            TRAINABLE_EMB = True
            #embedding_path = os.path.join(MODELS_DIR, dataset_dir_name, 'type{}'.format(str(3)), 'embedding', 'activity-role-embedding.npy')
            embedding_path = None
        predictive_model = Transformer(NUM_LAYERS, d_model, num_heads, NUM_UNIT_DFF,
                target_vocab_size, PE, embedding_path, TRAINABLE_EMB, rate=0.1)

        # training utils
        learning_rate = CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
            epsilon=1e-9)
        ckpt = tf.train.Checkpoint(predictive_model=predictive_model,
            optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=1)

        # training
        train_step_signature = [
                tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        val_loss = tf.keras.metrics.Mean(name='train_loss')
        val_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        @tf.function(input_signature=train_step_signature)
        def train_step(tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            
            mask = create_mask(tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = predictive_model(tar_inp, True, mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, predictive_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, predictive_model.trainable_variables))

            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))

        def val_step(tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            
            mask = create_mask(tar_inp)

            predictions, _ = predictive_model(tar_inp, False, mask)
            loss = loss_function(tar_real, predictions)
            val_loss(loss)
            val_accuracy(accuracy_function(tar_real, predictions))

        # training loop
        total_cases = df['case-id'].unique()
        total_dl_activities = []
        total_dl_roles = []
        for k in range(NUM_FOLDS):
            # define model (pe_target=max positional encoding)
            tf.keras.backend.clear_session()
            random.shuffle(total_cases)
            train_case = total_cases[
                    :int(np.round(PERC_TRAIN*len(total_cases)))]
            val_case = total_cases[
                    int(np.round(PERC_TRAIN*len(total_cases))):int(np.round((PERC_VAL+PERC_TRAIN)*len(total_cases)))]
            df_train = df[df['case-id'].isin(train_case)]
            df_val = df[df['case-id'].isin(val_case)]
            train_seq_creator = CreateSequences(df_train)
            generator_train = train_seq_creator.generate_data(BATCH_SIZE)
            val_seq_creator = CreateSequences(df_val)
            generator_val = val_seq_creator.generate_data(len(val_case))
            #generator_val = val_seq_creator.generate_data(BATCH_SIZE)
            num_batch_val = np.round(len(val_case))
            old_val_loss = 0
            patience = 0
            pbar = tqdm(range(EPOCHS))
            for epoch in pbar:

                train_loss.reset_states()
                train_accuracy.reset_states()

                for batch, tar in enumerate(generator_train):
                    train_step(tf.convert_to_tensor(tar))

                    if batch == steps_per_epoch:
                        break

#                for num_batch in range(num_batch_val): 
#                    val_step(tf.convert_to_tensor(next(generator_val)))
                val_step(tf.convert_to_tensor(next(generator_val)))
                if (epoch == 0 or (val_loss.result() < old_val_loss and epoch > 0)):
                    old_val_loss = val_loss.result()
                    ckpt_save_path = ckpt_manager.save()
                else:
                    patience += 1
                
                pbar.set_postfix({'Fold': k+1,
                                  'Training loss': np.around(train_loss.result(), 5),
                                  'Validation loss': np.around(val_loss.result(), 5),
                                  'Training accuracy' : np.around(train_accuracy.result(), 5),
                                  'Validation accuracy' : np.around(val_accuracy.result(), 5)})

                if patience > MAX_PATIENCE:
                    break

            # restore the best model
            ckpt.restore(ckpt_manager.latest_checkpoint)
            fold_model_path = os.path.join(model_path, 'fold{}'.format(k+1))
            if not os.path.exists(fold_model_path):
                os.mkdir(fold_model_path)
            predictive_model.save_weights(os.path.join(fold_model_path, 'weights'))
            with open(os.path.join(fold_model_path, 'train_indeces.pkl'), 'wb') as file:
                pickle.dump(train_case, file)
            with open(os.path.join(fold_model_path, 'val_indeces.pkl'), 'wb') as file:
                pickle.dump(val_case, file)

            # generate sequences
            total_val_cases = df_val['case-id'].unique()
            max_length_log = max(df_val.groupby(['case-id']).count()['activity-role-index'].values)
            #num_cases = len(total_val_cases)
            num_cases = 256
            start = np.ones((num_cases, 1), dtype=int)
            for gen_method in ['argmax', 'random']:
                output = tf.convert_to_tensor(start)
                print('------ {} ------'.format(gen_method.upper()))
                for i in range(max_length_log):
                    mask = create_mask(output)
                    # predictions shape == (batch_size, seq_len, vocab_size)
                    predictions, attention_weights = predictive_model(output, False, mask)
                    # select the last word from the seq_len dimension
                    predictions = predictions[:, -1:, :]
                    if gen_method == 'argmax':
                        predicted_id = tf.argmax(predictions, axis=2)
                    else:
                        predicted_id = tf.random.categorical(predictions[:, 0, :], 1)
                    # concat to the previous sequence
                    output = tf.concat([output, predicted_id], axis=-1)
                # to numpy
                output = output.numpy()

                # compute DL distance for every case in the test log
                # and associate to one recreated
                results_act = []
                results_role = []
                mean_dl_act_best = 0
                mean_dl_role_best = 0
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
                print('Computing DL on activities and roles')
                for case in tqdm(total_val_cases):
                    seq_activity = [1]
                    seq_activity.extend(df_val.loc[df_val['case-id']==case, 'activity-index'].values)
                    seq_activity.extend([2])
                    seq_role = [0]
                    seq_role.extend(df_val.loc[df_val['case-id']==case, 'role'].values)
                    seq_role.extend([0])
                    dl_act_best = 0
                    dl_role_best = 0
                    for i in range(output.shape[0]):
                        if 2 in output[i, :]:
                            first_end = np.where(output[i, :] ==  2)[0][0]
                        else:
                            first_end = output[i, :].shape[0]-1
                        gen_seq = list(output[i, :first_end+1])
                        gen_activity = list(map(lambda x: activities_map[x], gen_seq))
                        gen_role = list(map(lambda x: roles_map[x], gen_seq))
                        dl_act = DL(seq_activity, gen_activity)
                        dl_role = DL(seq_role, gen_role)
                        if i == 0:
                            dl_act_best = dl_act
                            dl_role_best = dl_role
                            cont_act = i
                            cont_role = i
                        elif dl_act > dl_act_best:
                            dl_act_best = dl_act
                            cont_act = i
                        elif dl_role > dl_role_best:
                            dl_role_best = dl_role
                            cont_role = i
                    results_act.append((case, dl_act_best, cont_act))
                    results_role.append((case, dl_role_best, cont_role))
                    mean_dl_act_best += dl_act_best
                    mean_dl_role_best += dl_role_best

                mean_dl_act_best /= len(total_val_cases)
                mean_dl_role_best /= len(total_val_cases)
                print(f'Mean Damerau Levenshtein metric on activities validation set: {mean_dl_act_best:.4f}')
                print(f'Mean Damerau Levenshtein metric on roles validation set: {mean_dl_role_best:.4f}')
                results_df = results_df.append({'hash': str(hash_code), 'dataset': dataset_dir_name, 'fold': k, 
                                                'embedding_type': embedding_type, 'trainable_embedding': TRAINABLE_EMB,
                                                'generation_type': gen_method, 'units': NUM_UNIT_DFF, 
                                                'num_heads': num_heads, 'd_model': d_model,
                                                'layers': NUM_LAYERS, 'batch_size': BATCH_SIZE,
                                                'dl_activities': mean_dl_act_best, 'dl_roles': mean_dl_role_best,
                                                'train_loss': train_loss.result().numpy(), 
                                                'val_loss': val_loss.result().numpy(), 
                                                'train_accuracy' :train_accuracy.result().numpy(), 
                                                'val_accuracy': val_accuracy.result().numpy(),
                                                'epochs_done': epoch, 'max_epochs': EPOCHS,
                                                'patience': patience, 'max_patience': MAX_PATIENCE}, ignore_index=True)

        for gen_method in ['argmax', 'random']:
            print('------ {} ------'.format(gen_method.upper()))
            data_mean = results_df.loc[results_df['generation_type'] == gen_method, ['dl_activities', 'dl_roles']].mean()
            act_mean = data_mean['dl_activities']
            print(f'K-fold mean Damerau Levenshtein metric on activities validation set: {act_mean:.4f}')
            role_mean = data_mean['dl_roles']
            print(f'K-fold mean Damerau Levenshtein metric on roles validation set: {role_mean:.4f}')

results_df.to_pickle(df_dir)
