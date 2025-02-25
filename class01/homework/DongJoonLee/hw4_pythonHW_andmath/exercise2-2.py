import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 4*x +6


def grad_fx(x):
    return 2*x - 4

NumberOfPoints = 101
x = np.linspace(-5., 5, NumberOfPoints)
fx = f(x)




xid = np.argmin(fx)
xopt = x[xid]
print(xopt, f(xopt) )

plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(xopt, f(xopt), 'xr')

plt.show()