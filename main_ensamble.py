import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm

ds_train = tfds.load('helpdesk', split='train[:70%]')
ds_vali = tfds.load('helpdesk'split=

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

padded_ds = ds_train.padded_batch(32, 
                                  padded_shapes=padded_shapes,
                                  padding_values=padding_values).prefetch(tf.data.AUTOTUNE)


vocabulary = ['<START>',  '<END>','Resolve SW anomaly', 'Resolve ticket', 'RESOLVED', 
              'DUPLICATE', 'Take in charge ticket', 'Create SW anomaly',
              'Schedule intervention', 'VERIFIED', 'Closed', 'Wait',
              'Require upgrade', 'Assign seriousness', 'Insert ticket', 'INVALID']

layer_string_lookup = tf.keras.layers.StringLookup(
    vocabulary=vocabulary, num_oov_indices=1
)

NUM_ACT = 17
emb_dim_act = 25
emb_dim_res = 36
emb_dim_prod = 10
emb_dim_cost = int(np.round(np.power(398, 1/4)))
emb_dim_resp = 4
emb_dim_serv_lev = 2
emb_dim_serv_type = 2
emb_dim_workgroup = 2
emb_dim_seriousness = 2
emb_dim_variant = int(np.round(np.power(227, 1/4)))
emb_dim_day_part = 2
emb_dim_week_day = 2

emb_act = tf.keras.layers.Embedding(input_dim=NUM_ACT, output_dim=emb_dim_act)
emb_res = tf.keras.layers.Embedding(input_dim=24, output_dim=emb_dim_res)
emb_prod = tf.keras.layers.Embedding(input_dim=23, output_dim=emb_dim_prod)
emb_cost = tf.keras.layers.Embedding(input_dim=398, output_dim=emb_dim_cost)
emb_resp = tf.keras.layers.Embedding(input_dim=9, output_dim=emb_dim_resp)
emb_serv_lev = tf.keras.layers.Embedding(input_dim=6, output_dim=emb_dim_serv_lev)
emb_serv_type = tf.keras.layers.Embedding(input_dim=6, output_dim=emb_dim_serv_type)
emb_workgroup = tf.keras.layers.Embedding(input_dim=6, output_dim=emb_dim_workgroup)
emb_seriousness = tf.keras.layers.Embedding(input_dim=6, output_dim=emb_dim_seriousness)
emb_variant = tf.keras.layers.Embedding(input_dim=227, output_dim=emb_dim_variant)
emb_day_part = tf.keras.layers.Embedding(input_dim=3, output_dim=emb_dim_day_part)
emb_week_day = tf.keras.layers.Embedding(input_dim=8, output_dim=emb_dim_week_day)
time_normalization = tf.keras.layers.Normalization(axis=None)
time_normalization.adapt(padded_ds.map(lambda x: x["relative_time"]))

#input_act = tf.keras.layers.Input(shape=(None,))
#input_res = tf.keras.layers.Input(shape=(None,))
#input_prod = tf.keras.layers.Input(shape=(None,))
#input_cost = tf.keras.layers.Input(shape=(None,))
#input_resp = tf.keras.layers.Input(shape=(None,))
#input_serv_lev = tf.keras.layers.Input(shape=(None,))
#input_serv_type = tf.keras.layers.Input(shape=(None,))
#input_workgroup = tf.keras.layers.Input(shape=(None,))
#input_seriousness = tf.keras.layers.Input(shape=(None,))
#input_variant = tf.keras.layers.Input(shape=(None,))
#input_day_part = tf.keras.layers.Input(shape=(None,))
#input_week_day = tf.keras.layers.Input(shape=(None,))
#input_time = tf.keras.layers.Input(shape=(None,))

#act = emb_act(input_act)
#res = emb_res(input_res)
#prod = emb_prod(input_prod)
#cost = emb_cost(input_cost)
#resp = emb_resp(input_resp)
#serv_lev = emb_serv_lev(input_serv_lev)
#serv_type = emb_serv_type(input_serv_type)
#workgroup = emb_workgroup(input_workgroup)
#seriousness = emb_seriousness(input_seriousness)
#variant = emb_variant(input_variant)
#relative_time = tf.expand_dims(input_time, axis=-1)
#day_part = emb_day_part(input_day_part)
#week_day = emb_week_day(input_week_day)
#time_normalized = time_normalization(input_time)

#output = tf.keras.layers.Concatenate(axis=-1)([
#    act, res, prod, cost, resp, serv_lev, serv_type, workgroup,
#    seriousness, variant, relative_time, day_part, week_day
#])
#
#model = tf.keras.Model(inputs=[
#    input_act, input_res, input_prod, input_cost, input_resp,
#    input_serv_lev, input_serv_type, input_workgroup, input_seriousness,
#    input_variant, input_time, input_day_part, input_week_day],
#                       outputs=output)
#
#example = padded_ds.take(1).get_single_element()
#
#act_input = layer_string_lookup(example['activity'])
#out = model([
#    act_input, example['resource'], example['product'],
#    example['customer'], example['responsible_section'], example['service_level'],
#    example['service_type'], example['workgroup'], example['seriousness'],
#    example['variant'], example['relative_time'], example['day_part'], example['week_day']
#])

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangule, countin from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, 
                key_dim=embed_dim)
        self.ffn = keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(
                batch_size, seq_len, seq_len, tf.bool)
        attn_output = self.att(inputs, inputs, 
                attention_mask=causal_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_model():
    input_act = tf.keras.layers.Input(shape=(None,))
    input_res = tf.keras.layers.Input(shape=(None,))
#    input_prod = tf.keras.layers.Input(shape=(None,))
#    input_cost = tf.keras.layers.Input(shape=(None,))
#    input_resp = tf.keras.layers.Input(shape=(None,))
#    input_serv_lev = tf.keras.layers.Input(shape=(None,))
#    input_serv_type = tf.keras.layers.Input(shape=(None,))
#    input_workgroup = tf.keras.layers.Input(shape=(None,))
#    input_seriousness = tf.keras.layers.Input(shape=(None,))
#    input_variant = tf.keras.layers.Input(shape=(None,))
    input_day_part = tf.keras.layers.Input(shape=(None,))
    input_week_day = tf.keras.layers.Input(shape=(None,))
    input_time = tf.keras.layers.Input(shape=(None,))

    act = emb_act(input_act)
    res = emb_res(input_res)
#    prod = emb_prod(input_prod)
#    cost = emb_cost(input_cost)
#    resp = emb_resp(input_resp)
#    serv_lev = emb_serv_lev(input_serv_lev)
#    serv_type = emb_serv_type(input_serv_type)
#    workgroup = emb_workgroup(input_workgroup)
#    seriousness = emb_seriousness(input_seriousness)
#    variant = emb_variant(input_variant)
    time_normalized = time_normalization(input_time)
    relative_time = tf.expand_dims(time_normalized, axis=-1)
    day_part = emb_day_part(input_day_part)
    week_day = emb_week_day(input_week_day)
    
#    x = tf.keras.layers.Concatenate(axis=-1)([
#        act, res, prod, cost, resp, serv_lev, serv_type, workgroup,
#        seriousness, variant, relative_time, day_part, week_day
#    ])
    x = tf.keras.layers.Concatenate(axis=-1)([
        act, res, day_part, week_day, relative_time
    ])

    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    outputs_act = layers.Dense(NUM_ACT)(x)
    model = keras.Model(
            inputs=[input_act, input_res, input_day_part, input_week_day, input_time], 
            outputs=outputs_act)
    return model

# train loop
epochs = 100
embed_dim = emb_dim_act + emb_dim_res + emb_dim_day_part + emb_dim_week_day + 1
num_heads = 8
feed_forward_dim = 1024
model = create_model()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        ]
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

@tf.function(input_signature=train_step_signature)
def train_step(act, res, day_part, week_day, rel_time):
    input_data = [act[:, :-1], res[:, :-1], day_part[:, :-1], week_day[:, :-1], rel_time[:, :-1]]
    target_data = act[:, 1:]
    with tf.GradientTape() as tape:
        logits = model(input_data, training=True) 
        loss_value = loss_function(target_data, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_loss(loss_value)
    train_accuracy(accuracy_function(target_data, logits))

#bar_epoch = tqdm(range(epochs), desc='outer', position=0)
bar_epoch = tqdm(range(epochs), desc='Train', position=0)
bar_batch = tqdm(padded_ds, desc='Epoch', position=1, leave=False)
for epoch in tqdm(range(epochs), desc='Train', position=0):
    for step, batch_data in enumerate(tqdm(padded_ds, desc='Epoch', position=1, leave=False)):
        # extract data
        act_input = layer_string_lookup(batch_data['activity'])
        train_step(act_input, batch_data['resource'], batch_data['day_part'], batch_data['week_day'], batch_data['relative_time'])
        #bar_batch.update()
        #bar_batch.set_description(loss= train_loss.result().numpy(), accuracy = train_accuracy.result().numpy())
    #bar_epoch.set_postfix({'Loss': train_loss.result().numpy(), 'Accuracy': train_accuracy.result().numpy()})
    #bar_epoch.update()
        #bar_batch.update()
    #bar_batch.reset()
    #bar_epoch.update()
#bar_epoch.close()
#bar_batch.close()
#        input_data = [act_input[:, :-1], batch_data['resource'][:, :-1], batch_data['product'][:, :-1],
#                batch_data['customer'][:, :-1], batch_data['responsible_section'][:, :-1], 
#                batch_data['service_level'][:, :-1], batch_data['service_type'][:, :-1], batch_data['workgroup'][:, :-1], 
#                batch_data['seriousness'][:, :-1], batch_data['variant'][:, :-1], batch_data['relative_time'][:, :-1], 
#                batch_data['day_part'][:, :-1], batch_data['week_day'][:, :-1]]
#        input_data = [act_input[:, :-1], batch_data['resource'][:, :-1], batch_data['day_part'][:, :-1], batch_data['week_day'][:, :-1]]
#        target_data = act_input[:, 1:]
#        with tf.GradientTape() as tape:
#            logits = model(input_data, training=True) 
#            loss_value = loss_function(target_data, logits)
#
#        if tf.math.is_nan(loss_value):
#            breakpoint()
#
#        grads = tape.gradient(loss_value, model.trainable_weights)
#        optimizer.apply_gradients(zip(grads, model.trainable_weights))

#        if step % 50 == 0:
#            print(f'Epoch {epoch + 1} Batch {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
#
print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
