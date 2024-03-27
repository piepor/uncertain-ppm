import tensorflow as tf
import pickle as pkl

def bpic2013_utils(tfds_id):
    padded_shapes = {
        'activity': [None],
        'resource': [None],
        'impact': [None],
        'relative_time': [None],
        'day_part': [None],
        'week_day': [None],
        'case_id': [None],
    }
    padding_values = {
        'activity': '<PAD>',
        'resource': '<PAD>',
        'impact': '<PAD>',
        'relative_time': tf.cast(0, dtype=tf.int32),
        'day_part': tf.cast(0, dtype=tf.int64),
        'week_day': tf.cast(0, dtype=tf.int64),
        'case_id': '<PAD>',
    }
    if tfds_id:
        padded_shapes['tfds_id'] = ()
        padding_values['tfds_id'] = '<PAD>'

    with open(f'./data/vocabulary_act_BPIC13.pkl', 'rb') as file:
        vocabulary_act = pkl.load(file)

    with open(f'./data/vocabulary_res_BPIC13.pkl', 'rb') as file:
        vocabulary_res = pkl.load(file)

    vocabulary_impact = ['Low', 'Medium', 'High', 'Major']

    vocabularies = {'activity': vocabulary_act, 'resource': vocabulary_res, 'impact': vocabulary_impact}

    features_type = {'org:group': 'categorical', 'impact': 'categorical',
                     'day_part': 'categorical', 'week_day': 'categorical'}

    features_variant = {'impact': 'trace'}

    #return padded_shapes, padding_values, vocabulary_act, vocabulary_res, features_type, features_variant
    return padded_shapes, padding_values, vocabularies, features_type, features_variant
