import tensorflow as tf
from tensorflow.python.keras import layers
import UtilsNetwork as utils
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
from keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.python.keras.initializers import VarianceScaling
import pprint
import scipy
from termcolor import colored
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system('color')


class SetNetworkInfo:
    def __init__(self,
                 epochs,
                 batch_size,
                 n_input,
                 validation_size,
                 hidden_layers,
                 neurons_hidden_layer,
                 optimizer="adam",
                 activation="relu",
                 learning_rate=0.01,
                 repetition=5,
                 loss_function="mse",
                 selection="validation_loss",
                 kernel_regularizer="L2",
                 regularization_parameter=0.0,
                 dropout_value=0.0,
                 output_activation = None):

        self.epochs = epochs
        self.batch_size = batch_size
        self.n_input = n_input
        self.validation_size = validation_size
        self.hidden_layers = hidden_layers
        self.neurons_hidden_layer = neurons_hidden_layer
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.repetition = repetition
        self.loss_function = loss_function
        self.selection = selection
        self.kernel_regularizer = kernel_regularizer
        self.regularization_parameter = regularization_parameter
        self.dropout_value = dropout_value
        self.output_activation = output_activation

    def print_info(self):
        dict_info = {"epochs": self.epochs,
                     "batch_size": self.batch_size,
                     "n_input": self.n_input,
                     "validation_size": self.validation_size,
                     "hidden_layers": self.hidden_layers,
                     "neurons_hidden_layer": self.neurons_hidden_layer,
                     "activation": self.activation,
                     "optimizer": self.optimizer,
                     "learning_rate": self.learning_rate,
                     "repetition": self.repetition,
                     "loss_function": self.loss_function,
                     "selection": self.selection,
                     "kernel_regularizer": self.kernel_regularizer,
                     "regularization_parameter": self.regularization_parameter,
                     "dropout_value": self.dropout_value,
                     "output_activation": self.output_activation}
        print(colored("Network Information:", 'cyan', attrs=['bold']))
        pprint.pprint(dict_info)
        return dict_info


class BuildNetwork:
    def __init__(self,
                 network_info,
                 X,
                 y):
        self.network_info = network_info
        self.X = X
        self.y = y

    def assemble_network_structure(self):
        n_input = self.network_info.n_input
        width = self.network_info.hidden_layers
        height = self.network_info.neurons_hidden_layer
        kernel_regularizer = self.network_info.kernel_regularizer
        regularization_param = self.network_info.regularization_parameter
        activation = self.network_info.activation
        loss = self.network_info.loss_function
        learning_rate = self.network_info.learning_rate
        optimizer = self.network_info.optimizer
        output_activation = self.network_info.output_activation
        if optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        dropout_value = self.network_info.dropout_value
        kernel_reg = tf.keras.regularizers.l2(regularization_param)
        if kernel_regularizer == "L1":
            kernel_reg = tf.keras.regularizers.l1(regularization_param)

        seed_random_number(42)
        model = tf.keras.Sequential()
        model.add(layers.Dense(height,
                               activation=activation,
                               input_shape=(n_input,),
                               kernel_regularizer=kernel_reg,
                               kernel_initializer=VarianceScaling(scale=2, distribution="truncated_normal", mode="fan_in")
                               ))
        for i in range(width):
            model.add(layers.Dense(height,
                                   kernel_regularizer=kernel_reg,
                                   activation=activation,
                                   kernel_initializer=VarianceScaling(scale=2, distribution="truncated_normal", mode="fan_in")
                                   ))
            model.add(layers.Dropout(dropout_value))
        model.add(layers.Dense(1, activation=output_activation))
        model.compile(optimizer=optimizer, loss=loss)
        # kernel_initializer=variance_scaling_initializer(factor=2, mode="FAN_IN", uniform=True),
        # kernel_initializer=VarianceScaling(scale=2, distribution="uniform",mode="fan_in"),
        return model

    def train_network(self, verbose=0, seed=None):
        seed_random_number(42)
        X_train, X_val, y_train, y_val = train_test_split(self.X,
                                                          self.y,
                                                          random_state=42,
                                                          test_size=self.network_info.validation_size,
                                                          shuffle=True)
        shape0 = X_train.shape[0]
        # callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)]

        best_model = None
        best_error = None
        # best_index = None

        fig = plt.figure()
        for n_run in range(self.network_info.repetition):
            print(colored("*********************************************", 'cyan', attrs=['bold']))
            print(colored("Repetition number " + str(n_run), 'cyan', attrs=['bold']))
            network_model = self.assemble_network_structure()
            fitting = network_model.fit(X_train,
                                        y_train,
                                        epochs=self.network_info.epochs,
                                        batch_size=self.network_info.batch_size,
                                        validation_data=(X_val, y_val),
                                        shuffle=True,
                                        verbose=verbose,
                                        # callbacks=callbacks
                                        )

            y_val_pred = network_model.predict(X_val)
            y_train_pred = network_model.predict(X_train)
            if self.network_info.validation_size != 0:
                y_val_pred = y_val_pred.reshape(-1,)
            y_train_pred = y_train_pred.reshape(-1, )

            error = self.compute_error(fitting, y_val_pred, y_val, y_train_pred, y_train)

            if self.network_info.validation_size == 0:
                story = fitting.history['loss']
            else:
                story = fitting.history['val_loss']

            if best_model is None or error < best_error:
                best_model = network_model
                best_error = error
                # best_index = n_run
                # best_weights = best_model.get_weights()
                print("New best score found: ", best_error)
                plt.plot(np.arange(len(story)), story, label="Repetition " + str(n_run))
                # plt.savefig("./"+str(shape0)+".png")
        return best_model, best_error

    def compute_error(self, fitting, y_val_pred, y_val, y_train_pred, y_train):
        error = 10
        print("Selection method: ", self.network_info.selection)
        if self.network_info.selection == "validation_loss":
            error = np.mean(fitting.history['val_loss'][-1:])
        elif self.network_info.selection == "variance_prediction_error":
            error = utils.compute_prediction_error_variance(y_val, y_val_pred, 2)
        elif self.network_info.selection == "mean_prediction_error":
            error = utils.compute_mean_prediction_error(y_val, y_val_pred, 2)
        elif self.network_info.selection == "train_loss":
            error = np.mean(fitting.history['loss'][-1:])
        elif self.network_info.selection == "wasserstein_train":
            error = scipy.stats.wasserstein_distance(y_train, y_train_pred)

        return error


def single_thread():
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def seed_random_number(seed):
    # see https://stackoverflow.com/a/52897216
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)


