import tensorflow as tf
import pickle as pkl

def road_traffic_fines_utils(tfds_id):
    padded_shapes = {
        'activity': [None],
        'resource': [None],
        'amount': [None],
        'article': [None],
        'points': [None],
        'totalPaymentAmount': [None],
        'relative_time': [None],
        'day_part': [None],
        'week_day': [None],
        'case_id': [None],
    }
    padding_values = {
        'activity': '<PAD>',
        'resource': '<PAD>',
        'amount': tf.cast(0, dtype=tf.float32),
        'article': '<PAD>',
        'points': tf.cast(0, dtype=tf.int32),
        'totalPaymentAmount': tf.cast(0, dtype=tf.float32),
        'relative_time': tf.cast(0, dtype=tf.int32),
        'day_part': tf.cast(0, dtype=tf.int64),
        'week_day': tf.cast(0, dtype=tf.int64),
        'case_id': '<PAD>'
    }
    if tfds_id:
        padded_shapes['tfds_id'] = []
        padding_values['tfds_id'] = '<PAD>'

    with open(f'./data/vocabulary_act_road_traffic_fines.pkl', 'rb') as file:
        vocabulary_act = pkl.load(file)

    with open(f'./data/vocabulary_res_road_traffic_fines.pkl', 'rb') as file:
        vocabulary_res = pkl.load(file)

    with open(f'./data/vocabulary_article_road_traffic_fines.pkl', 'rb') as file:
        vocabulary_art = pkl.load(file)

    vocabularies = {'activity': vocabulary_act, 'resource': vocabulary_res, 'article': vocabulary_art}

    features_type = {'org:group': 'categorical', 'article': 'categorical', 'points': 'continuous',
                     'amount': 'continuous', 'totalPaymentAmount': 'continuous', 'day_part': 'categorical', 'week_day': 'categorical'}

    features_variant = {'amount': 'trace', 'points': 'trace', 'article': 'trace'}

    #return padded_shapes, padding_values, vocabulary_act, vocabulary_res, features_type, features_variant
    return padded_shapes, padding_values, vocabularies, features_type, features_variant
