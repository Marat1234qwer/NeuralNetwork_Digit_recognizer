import numpy as np
import matplotlib.pyplot as plt
from my_sklearn.datasets import load_digits
import random

digits = load_digits()  # загружаем данные

X = digits.data
y = digits.target

weight_0_1 = np.fromfile('weight_0_1.b', float).reshape(20, 64)  # загружаем полученные веса после обучения
weight_1_2 = np.fromfile('weight_1_2.b', float).reshape(10, 20)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sigmoid_mapper = np.vectorize(sigmoid)


def predict(inputs):
    inputs_1 = np.dot(weight_0_1, inputs)
    outputs_1 = sigmoid_mapper(inputs_1)
    inputs_2 = np.dot(weight_1_2, outputs_1)
    outputs_2 = sigmoid_mapper(inputs_2)
    return outputs_2


def main():
    TEST_QUANTITY = 2  # количество необходимых проверок
    for _ in range(TEST_QUANTITY):
        digit_index = random.randint(1500, 1797)  # выбираем изображение для предсказания
        digit = X[digit_index]

        g = 0
        for _ in range(8):       # изображение в виде матрицы чисел
            print(digit[g:g+8])  # каждому числу соответствует пиксель в оттенках серого цвета(0-белый цвет, 15-черный)
            g += 8
        print('\nCorrect digit: "{}"\n'.format(y[digit_index]))  # правильное значение выбранной цифры
        one_percent = sum(predict(digit).tolist()) / 100
        sort_pred = predict(digit).tolist()
        sort_pred.sort()
        for j in range(3):  # вывод предсказания нейронной сети. % - предпочтения нейронной сети
            print('Predict digit: "{}" - {}% '.format(predict(digit).tolist().index(sort_pred[-1-j]), sort_pred[-1-j] / one_percent))

        plt.figure(figsize=(5, 5))  # вывод изображения
        plt.imshow(np.reshape(digit, (8, 8)), interpolation='nearest')
        plt.set_cmap('binary')
        plt.show()


if __name__ == '__main__':
    main()
