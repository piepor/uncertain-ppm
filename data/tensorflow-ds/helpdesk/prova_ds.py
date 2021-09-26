import helpdesk
import tensorflow_datasets as tfds
import tensorflow as tf

ds_train = tfds.load('helpdesk', split='train[:70%]')
ds_vali = tfds.load('helpdesk', split='train[70%:85%]')
ds_test = tfds.load('helpdesk', split='train[85%:]')
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
padded_ds = ds_train.padded_batch(16, 
                                  padded_shapes=padded_shapes,
                                  padding_values=padding_values)
#batched_ds = ds.batch(16)

for element in ds_train:
    act = element['activity']
    res = element['resource']
    cost = element['customer']
    resp = element['responsible_section']
    serv_lev = element['service_level']
    serv_type = element['service_type']
    seri = element['seriousness']
    wg = element['workgroup']
    var = element['variant']
    time = element['relative_time']
    day_part = element['day_part']
    week_day = element['week_day']
    print(act)
    print(time)

for batch in padded_ds:
    act = batch['activity']
    res = batch['resource']
    cost = batch['customer']
    resp = batch['responsible_section']
    serv_lev = batch['service_level']
    serv_type = batch['service_type']
    seri = batch['seriousness']
    wg = batch['workgroup']
    var = batch['variant']
    time = batch['relative_time']
    day_part = batch['day_part']
    week_day = batch['week_day']
    print(act)
    print(time)
    print(day_part)
    print(week_day)
