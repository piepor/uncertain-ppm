import os
from preprocessing_utils import EventLogImporter, RolesDiscover
from EmbeddingModel import EmbeddingModel
import logging

#logging.basicConfig(level=logging.DEBUG)

#data_path = ['./data/Helpdesk.xes', './data/BPI_2012_W_complete.xes', './data/BPI_Challenge_2012.xes']
data_dir = './data'
model_dir = './models'
#data_sets = ['BPI_2012_W_complete.xes', 'BPI_Challenge_2012.xes']
data_sets = ['Helpdesk.xes', 'BPI_Challenge_2012.xes']
batch_size = 64
# model type:
# 1 - embeddings predicting role
# 2 - embeddings predicting role and activities at the same time
# 3 - embeddings predicting roles and then using them to initialize emebdding predicting activities
#type_models = [1, 2, 3]
type_models = [3]
embedding_dimension_multipliers = [1, 2, 4]

for data_set in data_sets:
    for type_model in type_models:
        for multiplier in embedding_dimension_multipliers:
            data_path = os.path.join(data_dir, data_set)
            data_dir_name = data_set.split('.')[0].lower().replace('_', '-')
            save_path = os.path.join(model_dir, data_dir_name, 'type{}'.format(type_model), 'embedding')
            if not os.path.exists(save_path):
                old_mask = os.umask(000)
                os.makedirs(save_path, mode=0o777)
                _ = os.umask(old_mask)

            log = EventLogImporter(data_path, 'log-normal')
            df = log.get_df()
            roles = RolesDiscover(df)
            df = roles.get_df()

            if type_model == 1 or type_model == 2:
                emb_model = EmbeddingModel(df, type_model, emb_dim_mul=multiplier)
                emb_model.train_embedding(save_path, batch_size)
                #emb_model.visualize_embedding(emb_model.model.get_layer('activity_role_embedding').get_weights()[0])
            elif type_model == 3:
                initial_emb_path = os.path.join(model_dir, data_dir_name, 'type1', 'embedding')
                emb_model = EmbeddingModel(df, type_model, initial_emb_path, emb_dim_mul=multiplier)
                emb_model.train_embedding(save_path, batch_size)
                #emb_model.visualize_embedding(emb_model.model.get_layer('activity_role_embedding').get_weights()[0])
