import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7 # 微小な値
    return -np.sum(t * np.log(y + delta))

# t[2]が正解
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# y[2]が一番大きな値(つまり正解)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# y[7]が一番大きな値(つまり不正解)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
