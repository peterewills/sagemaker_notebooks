# See https://github.com/aws/sagemaker-python-sdk#tensorflow-sagemaker-estimators for more detail on
# the required structure for this script. See mnist_dnn_clf.py for more comments on my interpretation.

# compare to sample-notebooks/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_layers/abalone.py

import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput


# it's important that this be called 'inputs'
INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001

params = {"learning_rate": LEARNING_RATE}

# this is the main function, defining our model to be used by the estimator
def model_fn(features, labels, mode, params):
    """Defines the model to be passed into the estimator as a tf.estimator.EstimatorSpec"""
    
    # establish layer structure:
    first_hidden_layer = tf.layers.dense(features[INPUT_TENSOR_NAME], 300, activation=tf.nn.relu)
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 100, activation=tf.nn.relu)
    logits = tf.layers.dense(second_hidden_layer, 10, activation=None)
    predictions = tf.argmax(logits,axis=1) # index equals digit label

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    # export_outputs tells it what outputs to save in a saved model. I don't know
    # what the PredictOutput() class does, exactly.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          predictions={"logits": logits},
                                          export_outputs={SIGNATURE_NAME : 
                                                          PredictOutput({"logits": logits})}
                                         )
    
    # below executes otherwise, which is to say
    # elif mode == tf.estimator.ModeKeys.TRAIN or mode == tf.Estimator.ModeKeys.EVAL
    
    # cross-entropy loss
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    loss = tf.reduce_mean(xentropy)
    
    # report accuracy as additional evaluation metric
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(predictions,labels)
    }
    
    # optimizer and training operation
    # optimizer = tf.train.AdagradOptimizer(learning_rate=params["learning_rate"])
    optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops = eval_metric_ops)


def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=[1, 784])
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()

# this syntax is different from in mnist_dnn_clf.py, but is more compact and clean
def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'mnist_train.csv')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'mnist_test.csv')


def _input_fn(training_dir, training_filename):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename), 
        target_dtype=np.int, 
        features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()