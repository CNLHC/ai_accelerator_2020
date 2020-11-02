# LeNet for MNIST using Keras and TensorFlow
import argparse
import os
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from datetime import datetime
from tensorflow import keras

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import datasets
import numpy as np
from keras import backend as K
import tensorflow as tf


# tf.debugging.set_log_device_placement(True)
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)


class LeNetMNIST():
    def __init__(self,
                 data_path="./data",
                 logs_path="./log",
                 output_path="./output"
                 ):

        self.data_path = data_path
        self.logs_path = logs_path
        self.output_path = output_path
        self.input_shape = (28, 28, 1)
        raw = np.load(os.path.join(data_path, "mnist.npz"))
        self.train_data = raw['x_train']
        self.train_labels = raw['y_train']
        self.test_data = raw['x_test']
        self.test_labels = raw['y_test']
        self.train_data = np.expand_dims(self.train_data, -1)
        self.test_data = np.expand_dims(self.test_data, -1)
        self.train_labels = to_categorical(self.train_labels, 10)
        self.test_labels = to_categorical(self.test_labels, 10)
        print("label shape", self.train_labels.shape)
        print("data shape", self.train_data.shape)
        self.model = None

    def _build(self):
        self.model = Sequential([
            keras.Input(shape=self.input_shape),
            layers.Conv2D(filters=20, kernel_size=(5, 5),
                          padding="same", input_shape=(28, 28, 1)),
            layers.Activation(activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(filters=50, kernel_size=(5, 5), padding="same"),
            layers.Activation(activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(500),
            layers.Activation(activation='relu'),
            layers.Dense(10),
            layers.Activation(activation='softmax'),
        ])

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(lr=0.01),
            metrics=["accuracy"]
        )

    def train(self):
        if self.model is None:
            self._build()
        logdir = os.path.join(
            self.logs_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        self.model.fit(
            self.train_data,
            self.train_labels,
            batch_size=128,
            epochs=40,
            validation_split=0.2,
            callbacks=[tensorboard_callback],
            verbose=1)
        self.model.save(self.output_path)

    def evaluate(self):
        if self.model is None:
            raise ValueError
        (loss, accuracy) = self.model.evaluate(
            self.test_data,
            self.test_labels,
            batch_size=128,
            verbose=1)

        print("accuracy:", accuracy)

    def load(self):
        self.model = keras.models.load_model("./model")


def __main__():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--train', '-t',  dest="train",
                        help='train model', action='store_true')
    parser.add_argument('--load', '-l', dest='load',
                        help='load model', action='store_true')

    def dir_path(path):
        if os.path.isdir(path) or os.path.isfile(path):
            return path
        else:
            raise argparse.ArgumentTypeError(
                f"readable_dir:{path} is not a valid path")

    parser.add_argument('--data_dir', '-x', dest='data_dir',
                        help='load model', type=dir_path, default="./data", required=False)
    parser.add_argument('--logs_dir', '-y', dest='logs_dir',
                        help='load model', type=dir_path, default="./log", required=False)
    parser.add_argument('--output_dir', '-z', dest='output_dir',
                        help='load model', type=dir_path, default="./output", required=False)

    args = parser.parse_args()
    obj = LeNetMNIST(
        data_path=args.data_dir,
        logs_path=args.logs_dir,
        output_path=args.output_dir
    )
    if args.load:
        obj.load()
    if args.train:
        obj.train()
    obj.evaluate()


if __name__ == "__main__":
    __main__()
