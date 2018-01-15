import csv
import numpy as np


def read_spect_dataset(train=True):
    if train:
        csvreader = csv.reader(open('spect/SPECT.train.csv'))
    else:
        csvreader = csv.reader(open('spect/SPECT.test.csv'))

    X = []
    y = []
    row = 0
    for line in csvreader:
        X += [[]]
        column = map(int, line)
        for i, value in enumerate(column):
            if i == 0:
                y.append(value)
            else:
                X[row].append(value)
        row += 1

    return np.array(X), np.array(y)


# read_spect_dataset()