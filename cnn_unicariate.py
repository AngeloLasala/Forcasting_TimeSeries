"""Univariate forcasting using smple CNN model"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from mlp_multisteps import split_sequences

if __name__ == '__main__':
    data1 = np.linspace(10,10000,1000)
    data2 = data1 + 10000

    n_steps_input = 5
    n_steps_output = 1
    X_train, y_train = split_sequences(data1, n_steps_input=n_steps_input)
    X_test, y_test = split_sequences(data2, n_steps_input=n_steps_input)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_steps_output)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_steps_output)
   
    #Model
    inputs = Input(shape=(n_steps_input,n_steps_output))
    conv = Conv1D(10, 3, activation='relu')(inputs)
    pool = MaxPooling1D(2)(conv)
    flatten = Flatten()(pool)
    dense = Dense(150, activation='relu')(flatten)
    outputs = Dense(1, activation='linear')(dense)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    print(model.summary()) 
    
    #Training
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    history = model.fit(X_train, y_train, validation_split=0.3, epochs=100, verbose=1)

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


    