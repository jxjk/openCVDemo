# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 30, num=50)
y = 0.2*x+[np.random.random() for _ in range(50)]
print(x.dtype)
print(x)
print(y.dtype)
print(y)
if __name__ == '__main__':
    plt.figure(figsize=(10, 5), facecolor='w')
    plt.plot(x, y, 'ro', lw=2, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel(u'X', fontsize=16)
    plt.ylabel(u'Y', fontsize=16)
    plt.show()