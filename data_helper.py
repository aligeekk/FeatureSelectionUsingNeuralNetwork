from csv import reader
import numpy as np
import math


FILENAME = 'magic04.data'
PREDICTOR_COL = [i for i in range(0, 10)]
LABEL_COL = 10


def split_data(data):
    np.random.shuffle(data)
    n_row = data.shape[0]
    training_ = data[:math.floor(0.5 * n_row), :]
    validation_ = data[math.ceil(0.5 * n_row):math.floor(0.8 * n_row), :]
    testing_ = data[math.ceil(0.8 * n_row):, :]
    # print(training_.shape)
    # print(validation_.shape)
    # print(testing_.shape)
    return training_, validation_, testing_
    

def get_file_data(filename=FILENAME):
    data = list()
    with open(filename, 'r') as f:
        file_reader = reader(f)
        for row in file_reader:
            r = list()
            for i in PREDICTOR_COL:
                r.append(float(row[i]))
            if row[10] == 'g':
                r.append(1.0)
            else:
                r.append(0.0)
            data.append(r)
    nd_array = np.array(data)
    # print(nd_array.shape)
    # print(nd_array[0])
    return nd_array


def main():
    """
    Entry point.
    :return:
    """
    data = get_file_data()
    split_data(data)


if __name__ == '__main__':
    main()

