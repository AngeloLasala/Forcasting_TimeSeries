"""Multi-steps time-series forcasting using MLP"""
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model


def split_sequences(sequence, n_steps_input, n_steps_output=1):
    """
    Split a time series in a data frame in wich each row is composed by n_steps_input data,
    the target is also multi-steps array defined by n_steps_output.
    The output is composed by the splited dataset X and the target class y

    Parameters
    ----------
    sequence: list, numpy array
        times series data to split, it could be a list or numpy array

    n_steps_input: integer
        number of time steps for each istance in input data

    n_steps_output: integer(optional)
        default 1. Number of time steps for output target.

    Results
    -------
    X: numpy array
        Arrey of the input set, its shape is (len(sequences)-n_steps_input, n_steps_input)

    y: numpy array
        Arrey of the input target, its shape is
        (len(sequences)-n_steps_input-n_steps_output, n_steps_output)
    """

    X = [sequence[i:i+n_steps_input] for i in range(len(sequence)-n_steps_input-n_steps_output)]
    y = [sequence[i+n_steps_input:i+n_steps_input+n_steps_output] for i in range(len(sequence)-n_steps_input-n_steps_output)]

    return np.array(X, dtype=object), np.array(y,dtype=object)

if __name__ == "__main__":

    data_train = np.linspace(10,900,90)
    data_test = data_train + 1000

    X_train, y_train = split_sequences(data_train, 4, n_steps_output=2)
    X_test, y_test = split_sequences(data_test, 4, n_steps_output=2)

    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    #Model
    inputs = Input(shape=(4,))
    hidden = Dense(300, activation='relu')(inputs)
    outputs = Dense(2, activation='linear')(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')

    #Training
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, verbose=1)

    #Prediction
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    y_pred = model.predict(X_test)

    for i,e in zip(y_pred, y_test):
        print(i, e)

    #Elbow curve
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.grid()
    plt.legend()
    plt.show()
