import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

ds_train = tfds.load('helpdesk', split='train[:70%]')

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
padded_ds = ds_train.padded_batch(2, 
                                  padded_shapes=padded_shapes,
                                  padding_values=padding_values)

vocabulary = ['<START>',  '<END>','Resolve SW anomaly', 'Resolve ticket', 'RESOLVED', 
              'DUPLICATE', 'Take in charge ticket', 'Create SW anomaly',
              'Schedule intervention', 'VERIFIED', 'Closed', 'Wait',
              'Require upgrade', 'Assign seriousness', 'Insert ticket', 'INVALID']

layer_string_lookup = tf.keras.layers.StringLookup(
    vocabulary=vocabulary, num_oov_indices=1
)

emb_act = tf.keras.layers.Embedding(input_dim=14, output_dim=4)
emb_res = tf.keras.layers.Embedding(input_dim=24, output_dim=4)
emb_prod = tf.keras.layers.Embedding(input_dim=23, output_dim=4)
emb_cost = tf.keras.layers.Embedding(input_dim=398, 
                                     output_dim=int(np.round(np.power(398, 1/4))))
emb_resp = tf.keras.layers.Embedding(input_dim=9, output_dim=4)
emb_serv_lev = tf.keras.layers.Embedding(input_dim=6, output_dim=2)
emb_serv_type = tf.keras.layers.Embedding(input_dim=6, output_dim=2)
emb_workgroup = tf.keras.layers.Embedding(input_dim=6, output_dim=2)
emb_seriousness = tf.keras.layers.Embedding(input_dim=6, output_dim=2)
emb_variant = tf.keras.layers.Embedding(input_dim=227,
                                        output_dim=int(np.round(np.power(227, 1/4))))
emb_day_part = tf.keras.layers.Embedding(input_dim=3, output_dim=2)
emb_week_day = tf.keras.layers.Embedding(input_dim=8, output_dim=2)

input_act = tf.keras.layers.Input(shape=(None,))
input_res = tf.keras.layers.Input(shape=(None,))
input_prod = tf.keras.layers.Input(shape=(None,))
input_cost = tf.keras.layers.Input(shape=(None,))
input_resp = tf.keras.layers.Input(shape=(None,))
input_serv_lev = tf.keras.layers.Input(shape=(None,))
input_serv_type = tf.keras.layers.Input(shape=(None,))
input_workgroup = tf.keras.layers.Input(shape=(None,))
input_seriousness = tf.keras.layers.Input(shape=(None,))
input_variant = tf.keras.layers.Input(shape=(None,))
input_day_part = tf.keras.layers.Input(shape=(None,))
input_week_day = tf.keras.layers.Input(shape=(None,))
input_time = tf.keras.layers.Input(shape=(None,))

act = emb_act(input_act)
res = emb_res(input_res)
prod = emb_prod(input_prod)
cost = emb_cost(input_cost)
resp = emb_resp(input_resp)
serv_lev = emb_serv_lev(input_serv_lev)
serv_type = emb_serv_type(input_serv_type)
workgroup = emb_workgroup(input_workgroup)
seriousness = emb_seriousness(input_seriousness)
variant = emb_variant(input_variant)
relative_time = tf.expand_dims(input_time, axis=-1)
day_part = emb_day_part(input_day_part)
week_day = emb_week_day(input_week_day)

output = tf.keras.layers.Concatenate(axis=-1)([
    act, res, prod, cost, resp, serv_lev, serv_type, workgroup,
    seriousness, variant, relative_time, day_part, week_day
])

model = tf.keras.Model(inputs=[
    input_act, input_res, input_prod, input_cost, input_resp,
    input_serv_lev, input_serv_type, input_workgroup, input_seriousness,
    input_variant, input_time, input_day_part, input_week_day],
                       outputs=output)

example = padded_ds.take(1).get_single_element()

act_input = layer_string_lookup(example['activity'])
out = model([
    act_input, example['resource'], example['product'],
    example['customer'], example['responsible_section'], example['service_level'],
    example['service_type'], example['workgroup'], example['seriousness'],
    example['variant'], example['relative_time'], example['day_part'], example['week_day']
])
