import numpy as np


class Digits(object):
    def __init__(self):
        self.data = np.loadtxt('digits_data.txt', float).reshape(1797, 64)
        self.target = np.loadtxt('digits_target.txt')


def load_digits():
    return Digits()
