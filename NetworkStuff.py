import tensorflow as tf
import numpy as np
import random as rand

def get_network_input(p):
    my_folders = [x[0] for x in os.walk(p.data_location)][1:]
    input_list = []
    label_list = []
    for folder in my_folders:
        mat, energy, _ = get_adjacency_mat_from_file(folder)
        mat = mat.flatten()
        n = (p.maxnum**2-mat.size)
        #pad zeroes to make all input have the same dimension
        mat = np.pad(mat, (0, n), mode='constant')
        #print(mat.shape)
        input_list.append(mat)
        label_list.append(energy)
    return np.array(input_list), np.array(label_list)

def make_model(p):

    my_input, my_labels = get_network_input(p)
    #print(my_input.shape)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(5, activation='relu', input_shape=my_input[0].shape))
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

def test():
    print("hello world, fghrthdfghrsgfhsfg")