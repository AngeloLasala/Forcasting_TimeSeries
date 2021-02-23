"""Simple MPL to forcast a time series"""
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense 
from keras.models import Model

def split_sequences(sequence, n_steps):
    """
    Split a time series in a data frame in wich each row is composed by n_steps data.

    Parameters
    ----------
    sequence: list, numpy array
        times series data to split, it coulnd be a list or numpy array

    n_steps: integer
        number of time steps for each istance

    Results
    -------
    X: numpy array
        Arrey of the input set, its shape is (len(sequences)-n_steps, n_steps) 

    y: numpy array
        Arrey of the input target, its shape is (len(sequences)-n_steps, 1)
    """

    X = [sequence[i:i+n_steps] for i in range(len(sequence)-n_steps)]
    y = [sequence[i+n_steps] for i in range(len(sequence)-n_steps)]
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    data_train = np.linspace(10,900,90)
    data_test = np.linspace(1000,1030,4)

    X_train, y_train = split_sequences(data_train,3)
    X_test, y_test = split_sequences(data_test, 3)

    print(X_train.shape)

    #Model
    inputs = Input(shape=(3,))
    hidden = Dense(500, activation='relu')(inputs) 
    outputs = Dense(1, activation='linear')(hidden) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())

    #Training
    history = model.fit(X_train ,y_train ,validation_split=0.3, epochs=300, verbose=1)

    #Prediction
    y_pred = model.predict(X_test)
    print(y_pred, y_test)


    #Elbow curve
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss') 
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.grid()
    plt.legend()
    plt.show()
