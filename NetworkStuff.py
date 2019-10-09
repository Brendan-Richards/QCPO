import tensorflow as tf
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import random as rand



def f(fourier_amps, fourier_freqs, x):
    num = 0
    for i in range(len(fourier_amps)):
        if rand.random() > 0.5:
            num += fourier_amps[i]*np.sin(fourier_freqs[i]*x)
        else:
            num += fourier_amps[i] * np.cos(fourier_freqs[i]*x)
    return num

def make_training_data(p):
    inputs = []
    labels = []

    times = np.linspace(0, p.tTotal, num=p.numt)

    for i in range(p.net_train_size):
        amps = []
        for j in range(p.num_controls):
            fourier_amps = []
            fourier_freqs = []
            for r in range(len(p.fourier_amps)):
                fourier_amps.append(rand.random()*p.fourier_max_amp)
                fourier_freqs.append(rand.random()*p.fourier_max_freq)
            amps.append([f(fourier_amps, fourier_freqs, x) for x in times])
        inputs.append(amps)
        print("made amplitudes: " + str(i))

    inputs = np.array(inputs)
    #print(np.array(inputs))

    for r in range(p.net_train_size):
        U = []
        x = []
        for i in range(p.numt):  # loop over time steps
            mat = p.H0  # matrix in the exponential
            for k in range(p.num_controls):  # loop over controls and add to hamiltonian
                mat = np.add(mat, inputs[r, k, i] * p.hks[k])

            mat = -1 * 1j * p.dt * mat
            # print(mat)
            U.append(expm(mat))  # do the matrix exponential
            if i == 0:
                x.append(U[i])
            else:
                x.append(np.matmul(U[i], x[i - 1]))
        labels.append(fidelity(p.target, x))
        print("computed label: " + str(r))
    labels = np.array(labels)

    flat_inputs = np.array([x.flatten().tolist() for x in inputs])

    return flat_inputs, labels


def fidelity(tgt, X):
    return HS(tgt, X[-1]) * HS(X[-1], tgt)

# Hilbert-Schmidt-Product of two matrices M1, M2
def HS(M1, M2):
    v1 = (np.matmul(M1.conjugate().transpose(), M2)).trace()
    return v1 / M1.shape[0]

def make_model(p):

    my_input, my_labels = make_training_data(p)
    #print(my_input.shape)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=my_input[0].shape))
    #model.add(tf.keras.layers.Dense(10, activation='relu'))
    #model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(p.learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    model.fit(my_input, my_labels,
              epochs=p.num_epochs,
              steps_per_epoch=p.num_steps_per_epoch)
    return model

def create_new_model(p):
    m = make_model(p)
    m.save("neural_net_models/1qubit_hadamard.h5")