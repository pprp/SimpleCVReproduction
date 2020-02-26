# 学号： 2016012963
# 函数：y = cos(20x+63)
import math
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return math.cos(20 * x + 63)

if __name__ == "__main__":
    x = np.linspace(-63.0 / 20, float(2 * math.pi - 63) / 20, 2000)
    y = [func(i) for i in x]

    f1 = np.polyfit(x, y, 3)
    p = np.poly1d(f1)

    y2 = p(x)

    plt.scatter(x, y)
    plt.scatter(x, y2)
    plt.show()