

import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras


def get_nn_model(input_dim):
    """
    Creation of the neural network for the Adult Task.
    :param input_dim: input dimension.
    :return:
    """
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(64, activation="tanh")(inputs)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(64, activation="tanh")(x)
    x = keras.layers.Dropout(0.1)(x)
    # output = layers.Dense(1, activation="sigmoid")(x)
    output = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
    return model


DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}'

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()
shadow_data = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)


# Creation of the model
model = get_nn_model(train_set.shape[1])

# Compilation of the model and training.
opt = tf.optimizers.Adam(learning_rate=0.005)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(train_set, train_label, epochs=10, batch_size=512)

# Performances on training set
train_prediction = model.predict(train_set)
train_prediction = np.argmax(train_prediction, axis=1)
report = classification_report(train_label, train_prediction, digits=3)
print(report)

# Performances on test set
test_prediction = model.predict(test_set)

print(test_prediction)
print(type(test_prediction))
print(test_prediction.shape)

test_prediction = np.argmax(test_prediction, axis=1)
report = classification_report(test_label, test_prediction, digits=3)
print(report)

model.save(f'./models/nn_keras_{DS_NAME}.keras')
