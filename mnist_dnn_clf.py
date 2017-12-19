import numpy as np
import os
import tensorflow as tf

# I learned that it's important that this be called "inputs"
INPUT_TENSOR_NAME = 'inputs'


# needs to have two arguments, even though params doesn't get used
def estimator_fn(run_config,params):
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[784])]
    clf = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                     hidden_units=[300,100],
                                     n_classes=10,
                                     config=run_config)
    return clf


# the "serving" input function is the input function for when our esimator is in "predict" 
# mode. Thus, it only takes "feature specs", which tells the estimator what input to expect.
# No loss is calculated, as there are no "true" labels given in this mode.
def serving_input_fn(params):
    """Returns input function that would feed the model during prediction"""
    feature_spec = {INPUT_TENSOR_NAME : tf.FixedLenFeature(dtype=tf.float32, shape=[784])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


# The "train" input function is for when the estimator is in train mode. The code given
# by the demo notebook has a private function so that you don't need to keep rewriting
# these input function things, but I'm omitting it for the sake of structural simplicity.
def train_input_fn(training_dir, params):
    """Returns input function that would feed the model during training"""
    
    # get training dataset as tf.Dataset object
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, 'mnist_train.csv'),
        target_dtype=np.int,
        features_dtype=np.float32)
    
    # make it a tf.estimator input function
    input_fn = \
    tf.estimator.inputs.numpy_input_fn(
        x = {INPUT_TENSOR_NAME : np.array(training_set.data)},
        y = np.array(training_set.target),
        # batch_size = 50, # breaks stuff :(
        num_epochs = None,
        shuffle = True)()
    
    return input_fn


# Input function for when estimator is in "evaluate" (i.e. testing) mode.
def eval_input_fn(training_dir, params):
    """Returns input function that would feed the model during evaluation"""
    
    # get training dataset as tf.Dataset object
    testing_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, 'mnist_test.csv'),
        target_dtype=np.int,
        features_dtype=np.float32)
    
    # make it a tf.estimator input function
    input_fn = \
    tf.estimator.inputs.numpy_input_fn(
        x = {INPUT_TENSOR_NAME : np.array(testing_set.data)},
        y = np.array(testing_set.target),
        num_epochs = None,
        shuffle = False)()
    
    return input_fn