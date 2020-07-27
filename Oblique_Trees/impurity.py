from collections import Counter
import numpy as np
import sys
import time


def frequencies(data, target_attr):
    if len(data) == 0:
        return []
    try:
        c = Counter(data[:, target_attr])
        n_records = float(len(data))
        return np.array([v/n_records for v in c.values()])
    except TypeError:
        sys.stderr.write("Please use Numpy arrays!")
        sys.exit()


def gini(data, target_attr):
    freq = frequencies(data, target_attr)
    fs = np.square(freq)
    return 1 - np.sum(fs)

def twoing(left_label, right_label):
    sum = 0
    huge_val = np.inf
    left_len, right_len, n = len(left_label), len(right_label), (len(left_label) + len(right_label))
    labels = list(left_label) + list(right_label)
    n_classes = np.unique(labels)
    if (left_len != 0 & right_len != 0):
        for i in n_classes:
            idx = np.where(left_label == i)[0]
            print(len(idx))
            li = (len(idx) / left_len)
            idx = np.where(right_label == i)[0]
            ri = (len(idx) / right_len)
            sum += (np.abs(li - ri))
        twoing_value = ((left_len / n) * (right_len / n) * np.square(sum))/4

    elif (left_len == 0):
        for i in n_classes:
            idx = np.where(right_label == i)[0]
            print(len(idx))
            ri = (len(idx) / right_len)
            sum += ri
        twoing_value = ((left_len / n) * (right_len / n) * np.square(sum))/4

    else:
        for i in n_classes:
            idx = np.where(left_label == i)[0]
            print(len(idx))
            li = (len(idx) / left_len)
            sum += li
        twoing_value = ((left_len / n) * (right_len / n) * np.square(sum))/4
    if twoing_value == 0:
        return (huge_val)
    else:
        return (1/twoing_value)

if __name__ == '__main__':

    x=twoing([1,1,1,1,2,2,2],[1,1,1,2,2,2,3,3,3,3,3])
    #[1,1,1,2,2,2,3,3,3,3,3]
    print(x)