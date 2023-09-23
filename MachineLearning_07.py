import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([
    [10, 50], [20, 30], [25, 30],
    [20, 60], [15, 70], [40, 40],
    [30, 45], [20, 45], [40, 30],
    [7, 35]
])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

mw1, ml1 = np.mean(x_train[y_train == 1], axis=0)
mw_1, ml_1 = np.mean(x_train[y_train == -1], axis=0)

sw1, sl1 = np.var(x_train[y_train == 1], axis=0)
sw_1, sl_1 = np.var(x_train[y_train == -1], axis=0)

print('ME: ', mw1, ml1, mw_1, ml_1)
print('Dispersion:', sw1, sl1, sw_1, sl_1)

x = [40, 10]


def a_1(x):
    return -(x[0] - ml_1) ** 2 / (2 * sl_1) - (x[1] - mw_1) ** 2 / (2 * sw_1)


def a1(x):
    return -(x[0] - ml1) ** 2 / (2 * sl1) - (x[1] - mw1) ** 2 / (2 * sw1)


y = np.argmax([a_1(x), a1(x)])

print('Class number (0 - caterpillar, 1 - ladybug): ', y)
