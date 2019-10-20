import os
import torch
import numpy as np
import scipy.misc as smp
import scipy.ndimage
from random import randint
import random

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_target(y_test, target_class = 13, width = 256, height = 256):
    
    y_target = y_test

    dilated_image = scipy.ndimage.binary_dilation(y_target[0, target_class, :, :], iterations=6).astype(y_test.dtype)

    for i in range(width):
        for j in range(height):
            y_target[0, target_class, i, j] = dilated_image[i,j]

    for i in range(width):
        for j in range(height):
            potato = np.count_nonzero(y_target[0,:,i,j])
            if (potato > 1):
                x = np.where(y_target[0, : ,i, j] > 0)
                k = x[0]
                #print("{}, {}, {}".format(i,j,k))
                if k[0] == target_class:
                    y_target[0,k[1],i,j] = 0.
                else:
                    y_target[0, k[0], i, j] = 0.

    return y_target

def generate_target_swap(y_test):


    y_target = y_test

    y_target_arg = np.argmax(y_test, axis = 1)

    y_target_arg_no_back = np.where(y_target_arg>0)

    y_target_arg = y_target_arg[y_target_arg_no_back]

    classes  = np.unique(y_target_arg)

    if len(classes) > 3:

        first_class = 0

        second_class = 0

        third_class = 0

        while first_class == second_class == third_class:
            first_class = classes[randint(0, len(classes)-1)]
            f_ind = np.where(y_target_arg==first_class)
            #print(np.shape(f_ind))

            second_class = classes[randint(0, len(classes)-1)]
            s_ind = np.where(y_target_arg == second_class)

            third_class = classes[randint(0, len(classes) - 1)]
            t_ind = np.where(y_target_arg == third_class)

            summ = np.shape(f_ind)[1] + np.shape(s_ind)[1] + np.shape(t_ind)[1]

            if summ < 1000:
                first_class = 0

                second_class = 0

                third_class = 0

        for i in range(256):
            for j in range(256):
                temp = y_target[0,second_class, i,j]
                y_target[0,second_class, i,j] = y_target[0,first_class,i,j]
                y_target[0, first_class,i, j] = temp


    else:
        y_target = y_test
        print('Not enough classes to swap!')
    return y_target
