

import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import cv2


def load_data(path):
    classes = os.listdir(path)
    data_set_total = np.array(np.ones((48, 48, 3))).reshape((1, 48, 48, 3))
    labels_total = np.ones((1, len(classes)))
    for i in range(len(classes)):
        print('i=', i)
        data_set = np.array([])
        labels = np.zeros(len(classes))

        alp = os.listdir(path+classes[i])
        for j in range(len(alp)):
            if j % 1000 == 0 and j != 0:
                print('j=', j)
            image = cv2.imread(path+classes[i]+'/' + alp[j])

            if image.shape == (48, 48, 3):
                if j == 0:
                    data_set = image
                    data_set = data_set.reshape(1, 48, 48, 3)
                    labels = one_hot_label(i, len(classes))
                else:

                    image = image.reshape(1, 48, 48, 3)
                    data_set = np.concatenate((data_set, image), axis=0)
                    labels = np.concatenate(
                        (labels, one_hot_label(i, len(classes))), axis=0)

        labels_total = np.concatenate((labels_total, labels), axis=0)

        data_set_total = np.concatenate((data_set_total, data_set), axis=0)
    return data_set_total, labels_total


def one_hot_label(labels, n_classes):
    encoder = OneHotEncoder()
    encoder.fit(np.array(np.arange(n_classes)).reshape(n_classes, 1))
    return encoder.transform([[labels]]).toarray()
