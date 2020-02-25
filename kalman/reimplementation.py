import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 100, 100)
a = 2
x_real = (a * t**2) / 2

x_gps = x_real + np.random.normal(0, 150, size=100)
x_gps_var = 150**2

plt.plot(t, x_real)
plt.plot(t, x_gps)

# 初始化
preds = [x_gps[0]]
x_pred = x_gps[0]

x_pred_var = 0
v_var = 50

for i in range(1, t.shape[0]):
    # 获取v的分布
    v = (x_real[i] - x_real[i - 1]) + np.random.normal(0, v_var)
    # 更新x_pred的粗估值和方差
    x_pred = x_pred + v
    x_pred_var += v_var**2
    # kalman滤波
    x_pred = x_pred * x_gps_var /(x_gps_var + x_pred_var) + \
             x_gps[i] * x_pred_var / (x_pred_var + x_gps_var)
    x_pred_var = (x_pred_var * x_gps_var) / (x_pred_var + x_gps_var)
    preds.append(x_pred)

plt.plot(t, preds)

plt.show()