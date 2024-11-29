from my_sklearn.datasets import load_digits
import sys
import numpy as np

digits = load_digits()  # загружаем данные

X = digits.data
y = digits.target

one_hot_labels = np.zeros((len(y), 10))  # формируем выходные нейроны

for i, l in enumerate(y):
    one_hot_labels[i][l] = 1
expected_predict = one_hot_labels

train = []  # формируем список для обучения 1500 изображений
for i in range(1500):
    data_insert = []
    data_insert.append(tuple(X[i].tolist()))
    data_insert.append(expected_predict[i])
    train.append(data_insert)


class DigitRecognizerNN(object):
    def __init__(self, learning_rate=0.1):
        self.weight_0_1 = np.fromfile('weight_0_1_random.b', float).reshape(20, 64)
        self.weight_1_2 = np.fromfile('weight_1_2_random.b', float).reshape(10, 20)
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])

    def sigmoid(self, x):  # активационная функция
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        inputs_1 = np.dot(self.weight_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)
        inputs_2 = np.dot(self.weight_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        return outputs_2

    def train(self, inputs, expected_predict):
        inputs_1 = np.dot(self.weight_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)  # активируем нейроны, присваиваем нейронам значение
        inputs_2 = np.dot(self.weight_1_2, outputs_1)  # умножаем вес на второй слой нейронов
        outputs_2 = self.sigmoid_mapper(inputs_2)  # активируем нейрон, присваиваем нейрону значение
        actual_predict = outputs_2

        error_layer_2 = np.array([actual_predict - expected_predict])  # ошибка, на сколько отличаются показания
        gradien_layer_2 = actual_predict * (1 - actual_predict)  # дифференциал функции
        weight_delta_layer_2 = error_layer_2 * gradien_layer_2  # числа на которые нужно откорректировать веса
        self.weight_1_2 = self.weight_1_2 - outputs_1 * weight_delta_layer_2.T * self.learning_rate  # новые веса последней нейронной связи

        error_layer_1 = np.dot(weight_delta_layer_2, self.weight_1_2)  # новое значение внутренних нейронов
        gradien_layer_1 = outputs_1 * (1 - outputs_1)  # дифференциал функции
        weight_delta_layer_1 = error_layer_1 * gradien_layer_1  # числа на которые нужно откорректировать веса первой нейронной связи
        self.weight_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weight_delta_layer_1).T * self.learning_rate  # новые веса первой нейронной связи


epochs = 200
learning_rate = 0.1

network = DigitRecognizerNN(learning_rate=learning_rate)

for e in range(epochs):  # вывод прогресса обучения
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.train(np.array(input_stat), correct_predict)
        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))
    predictions = network.predict(np.array(inputs_).T)
    diff = []
    for i in range(len(predictions)):
        index = correct_predictions[i].tolist().index(1)
        predictions[i][index]
        diff.append(correct_predictions[i][index] - predictions[i][index])
    train_loss = sum(diff) / len(diff)
    sys.stdout.write("\rProgress: {}%, Taining loss {}".format(str(100 * e/float(epochs))[:4], str(train_loss)[:5]))

# with open('weight_0_1.b', 'wb') as f:  # запись полученных весов в файлы
#     f.write(network.weight_0_1)
# with open('weight_1_2.b', 'wb') as f:
#     f.write(network.weight_1_2)
