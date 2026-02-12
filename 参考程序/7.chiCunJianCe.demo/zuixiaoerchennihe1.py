#from  paint_scatter_plot import x,y
import numpy as np 
import matplotlib.pyplot as plt

x1 = [0,0.61,1.22,1.83,2.44,3.06,3.67,4.28,4.89,5.5,6.12,6.73]
y1 = [0.15,0.88,0.36,1.31,1.48,0.64,1.13,1.49,1.13,1.33,1.35,1.83]
x = np.array(x1)
y = np.array(y1)
def Least_squares(x,y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(len(x)):
        k = (x[i]-x_)* (y[i]-y_)
        m += k
        p = np.square( x[i]-x_ )
        n = n + p
    a = m/n
    b = y_ - a* x_
    return a,b
if __name__ == '__main__':
    a,b = Least_squares(x,y)
    print(a,b)
    y1 = a * x + b
    plt.figure(figsize=(10, 5), facecolor='w')
    plt.plot(x, y, 'ro', lw=2, markersize=6)
    plt.plot(x, y1, 'r-', lw=2, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel(u'X', fontsize=16)
    plt.ylabel(u'Y', fontsize=16)
    plt.show()