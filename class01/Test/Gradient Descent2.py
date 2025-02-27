import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2 - 4*x + 6


NumberOfPoints = 101
x = np.linspace(-5, 5, NumberOfPoints)
fx = f(x)

xid = np.argmin(fx)
xopt = x[xid]
print(xopt, f(xopt))

# plt.plot(x,fx)
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')

# plt.plot(xopt, f(xopt), 'xr')
# plt.show()


def grad_fx(x):
    return 2*x - 4


def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxlter=10, verbose=True):
    paths = []
    for i in range(Maxlter):
        x1 = x0 - learning_rate * grad_func(x0)
        if verbose:
            print('{0:03d} : {1:4.3f}, {2:4.2E}'.format(i, x1, func(x1)))
        x0 = x1
        paths.append(x0)
        return(x0, func(x0, paths))


xopt, fopt, paths = steepest_descent(f, grad_fx, 0.0, learning_rate=1.2)

x = np.linspace(0.5, 2.5, 1000)
paths = np.array(paths)
plt.plot(x, f(x))