import tensorflow as tf
import yaml

def compute_features(file_path, vocabularies):
    with open(file_path, 'r') as file:
        features = list(yaml.load_all(file, Loader=yaml.FullLoader))
    for feature in features:
        if feature['feature-type'] == 'string':
            feature['vocabulary'] = vocabularies[feature['name']]
    return features

def loss_function(real, pred, loss_object):
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

def compute_input_signature(features):
    train_step_signature = []
    for feature in features:
        if feature['dtype'] == 'string':
            train_step_signature.append(tf.TensorSpec(shape=(None, None), dtype=tf.string))
        elif feature['dtype'] == 'int64': 
            train_step_signature.append(tf.TensorSpec(shape=(None, None), dtype=tf.int64))
        elif feature['dtype'] == 'int32': 
            train_step_signature.append(tf.TensorSpec(shape=(None, None), dtype=tf.int32))
    return train_step_signature
