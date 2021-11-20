import tensorflow as tf

def helpdesk_utils(tfds_id):
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
    if tfds_id:
        padded_shapes['tfds_id'] = ()
        padding_values['tfds_id'] = '<PAD>'

    vocabulary = ['<START>',  '<END>','Resolve SW anomaly', 'Resolve ticket', 'RESOLVED', 
                  'DUPLICATE', 'Take in charge ticket', 'Create SW anomaly',
                  'Schedule intervention', 'VERIFIED', 'Closed', 'Wait',
                  'Require upgrade', 'Assign seriousness', 'Insert ticket', 'INVALID']

    return padded_shapes, padding_values, vocabulary
