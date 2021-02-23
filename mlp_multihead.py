"""Multi-headed input MLP mode"""
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.utils import plot_model

def split_sequences(*sequences, target=None, n_steps=0):
    """
    Split a time series in a data frame in wich each row is composed by n_steps data.
    The number of time series is arbitrary because for each time we could be more than one
    features (attributes) but the target clas is univariate.
    Element of *sequences and target must be the same shape.

    Parameters
    ----------
    *sequences: list, numpy array
        times series data used to forcast the class label, it could be a list or numpy array.

    target: list, numpy array(optional)
       target class,  it could be a list or numpy array. if None the output is only the stack
       of the *sequences and it means there isn't a target

    n_steps: integer(optional)
        number of time steps for each istance, if 0 the array of shape (len(sequences),2)
        output[0]: np.array of *sequences at same time
        output[1]: np.array of target

    Results
    -------
    Xy_train: numpy array
        it is the input trainig set with shape (len(sequence)-n_steps, n_steps, len(*sequences))

    y: numpy array
        it is the input  target classset with shape (len(sequence)-n_steps, 1)
    """
    X_train = np.stack(sequences, axis=-1)
    y_train = target

    if (target is not None) and (n_steps == 0):
        Xy_train = [[X_train[i], y_train[i]] for i in range(len(y_train))]
        y_train = target

    if (target is not None) and (n_steps > 0):
        Xy_train = [np.array(X_train[i:i+n_steps]) for i in range(len(y_train)-n_steps)]
        y_train = [y_train[i+n_steps] for i in range(len(y_train)-n_steps)]

    if (target is None) and (n_steps == 0):
        Xy_train = X_train
        y_train = target

    if (target is None) and (n_steps > 0):
        Xy_train = [np.array(X_train[i:i+n_steps]) for i in range(len(X_train)-n_steps)]
        y_train = target

    return np.array(Xy_train, dtype=object), np.array(y_train)

if __name__ == "__main__":
    data1 = np.linspace(10,200,20)
    data2 = data1 + 5
    data3 = data1 + 200
    data4 = data3 + 5

    X_train, y_train = split_sequences(data1, data2, target=data1, n_steps=3)
    X_test, y_test = split_sequences(data3, data4, target=data3, n_steps=3)
    print(X_train[0],y_train[0])
    print(X_test[0],y_test[0])

    #Model
    input1 = Input(shape=(3,))
    hidden1 = Dense(200, activation='relu')(input1)
    input2 = Input(shape=(3,))
    hidden2 = Dense(200, activation='relu')(input2)
    merge = Concatenate()([hidden1, hidden2])
    outputs = Dense(1, activation='linear')(merge)
    model = Model(inputs=[input1, input2], outputs=outputs)
    model.compile(loss='mse', optimizer='adam')

    #Training
    X_train= np.asarray(X_train).astype(np.float32)
    history = model.fit([X_train[:,:,0],X_train[:,:,1]],y_train ,
                        validation_split=0.3, epochs=300, verbose=1)

    #Prediction
    X_test= np.asarray(X_test).astype(np.float32)
    y_pred = model.predict([X_test[:,:,1],X_test[:,:,1]])
    print(y_pred, y_test)

    #Elbow curve
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.grid()
    plt.legend()
    plt.show()
