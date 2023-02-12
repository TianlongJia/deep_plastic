# create class weight

import os
import numpy as np

def class_weights_4(path_train):
    num_classes=4
    folders = os.listdir(path_train)  # dir is your directory path
    num = []

    for folder in folders:
        dir = path_train + '/' + str(folder)
        onlyfiles = next(os.walk(dir))[2]  # dir is your directory path as string
        num.append(len(onlyfiles))

    total_img = np.sum(num)

    class_weight_train = {0: (total_img / num[0]) / num_classes,
                          1: (total_img / num[1]) / num_classes,
                          2: (total_img / num[2]) / num_classes,
                          3: (total_img / num[3]) / num_classes}

    return class_weight_train


def class_weights_binary(path_train):
    num_classes=2
    folders = os.listdir(path_train)  # dir is your directory path
    num = []

    for folder in folders:
        dir = path_train + '/' + str(folder)
        onlyfiles = next(os.walk(dir))[2]  # dir is your directory path as string
        num.append(len(onlyfiles))

    total_img = np.sum(num)

    class_weight_train = {0: (total_img / num[0]) / num_classes,
                          1: (total_img / num[1]) / num_classes,
                          }

    return class_weight_train