import numpy as np
from toy_models.envs.toy_functions import Simulator, TF2D_DEFAULT_CONFIG, minmax

frame_size = 1.15 # half frame side length
number_of_pixels = 80
pixel_size = (frame_size * 2) / number_of_pixels

config = TF2D_DEFAULT_CONFIG
simulator = Simulator(config)

def get_egg():
    x = np.linspace(10, 15, 500)
    y = np.linspace(10, 15, 500)
    xx, yy = np.meshgrid(x, y)
    X = xx.ravel()
    Y = yy.ravel()

    egg = simulator.run(X, Y)
    good = np.where(egg > np.exp(-2))[0]
    # exponential density reward -> np.exp(-2)
    X = minmax(xx.ravel(), (10,15), (-1,1))
    Y = minmax(yy.ravel(), (10,15), (-1,1))

    matrix = np.zeros((number_of_pixels, number_of_pixels))
    for i in good:
        idx_x = int((X[i] + frame_size) / pixel_size)
        idx_y = int((Y[i] + frame_size) / pixel_size)
        matrix[idx_x][idx_y] = True

    return matrix


def area_and_precision(states):
    egg = get_egg()
    
    area_matrix = np.zeros_like(egg)

    for s in states:
        idx_x = int((s[0] + frame_size) / pixel_size)
        idx_y = int((s[1] + frame_size) / pixel_size)

        area_matrix[idx_x, idx_y] = True

    new_shape = number_of_pixels * number_of_pixels
    egg = egg.reshape(new_shape)
    area_matrix = area_matrix.reshape(new_shape)

    good_pixels = sum(area_matrix * egg)
    area_rate = good_pixels / sum(egg)
    precision = good_pixels / sum(area_matrix)
    return area_rate, precision

 
            
