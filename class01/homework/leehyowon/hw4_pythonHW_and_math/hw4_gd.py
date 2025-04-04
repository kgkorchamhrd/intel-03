# -*- coding: utf-8 -*-
"""HW4_GD.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rq-yquP48u1JZRfSW3nS-Art755omZr_
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-2,2,11)
y=np.linspace(-2,2,11)

print(x)
print(y)

x,y=np.meshgrid(x, y)
print(x)
print(y)

f = lambda x,y: (x-1)**2+(y-1)**2
z=f(x,y)
print(z)

grad_f_x=lambda x,y:2*(x-1)
grad_f_y=lambda x,y:2*(y-1)

dz_dx=grad_f_x(x,y)
dz_dy=grad_f_y(x,y)

ax=plt.axes()
ax.contour(x,y,z,levels=np.linspace(0,10,20), cmap=plt.cm.jet)
ax.quiver(x,y,-dz_dx,-dz_dy)
ax.grid()
ax.axis('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()

def f(x):
  return x**2 - 4 * x +6
NumberOfPoints = 101
x = np.linspace(-5, 5, NumberOfPoints)
fx = f(x)
plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.show()

xid = np.argmin(fx)
xopt = x[xid]
print(xopt, f(xopt))

plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(xopt, f(xopt), 'xr')
plt.show()

def f(x):
    return x**2 -4*x + 6

def grad_fx(x):
    return 2*x - 4
def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxiter=10, verbose=True):
    paths = []
    for i in range(Maxiter):
        x1 = x0 -learning_rate * grad_func(x0)
        if verbose:
          print('{0:03d} : {1:4.3f}, {2:4.2E}'.format(i, x1, func(x1)))
        x0 = x1
        paths.append(x0)
    return(x0, func(x0), paths)
xopt, fopt, paths = steepest_descent(f, grad_fx, 0.0, learning_rate=1.2)
x = np.linspace(0.5, 2.5, 1000)
paths = np.array(paths)
plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(paths, f(paths), 'o-')
plt.show

plt.plot(f(paths), 'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of cost')
plt.show

xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=1)
x = np.linspace(0.5, 3.5, 1000)
paths = np.array(paths)
plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(paths, f(paths), 'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show

xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=0.001)
x = np.linspace(0.5, 3.5, 1000)
paths = np.array(paths)
plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths), 'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()

xpot, fopt, paths = steepest_descent(f, grad_fx, 3.0, learning_rate=0.9)
x=np.linspace(0.5, 3.5, 1000)
paths=np.array(paths)
plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(paths)
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from visualize import contour_with_quiver
from visualize import contour_with_path
from visualize import surf

xmin, xmax, xstep = -4.0, 4.0, .25
ymin, ymax, ystep = -4.0, 4.0, .25

x, y = np.meshgrid(np.arange(xmin, xmax +xstep, xstep),
                   np.arange(ymin, ymax +ystep, ystep))

f = lambda x,y: (x-1)**2+(y-1)**2
z=f(x,y)
minima = np.array([2, 2.])

f(*minima)

minima_ = minima.reshape(-1, 1)
print(minima, minima_)
surf(f, x, y, minima=minima_)

grad_f_x = lambda x, y: 2 * (x-2)
grad_f_y = lambda x, y: 2 * (y-2)

contour_with_quiver(f, x, y, grad_f_x, grad_f_y, minima=minima_)

def steepest_descent_twod(func, gradx, grady, x0, Maxiter=10,
                          learning_rate=0.25, verbose=True):
    paths = [x0]
    fval_paths = [f(x0[0], x0[1])]
    for i in range(Maxiter):
        grad = np.array([grad_f_x(*x0),grad_f_y(*x0)])
        x1 = x0 - learning_rate *grad
        fval = f(*x1)
        if verbose:
            print(i, x1, fval)
        x0 = x1
        paths.append(x0)
        fval_paths.append(fval)
    paths = np.array(paths)
    paths = np.array(np.matrix(paths).T)
    fval_paths = np.array(fval_paths)
    return(x0, fval, paths, fval_paths)

x0 = np.array([-2., -2])
xopt, fopt, paths, fval_paths = steepest_descent_twod(f, grad_f_x, grad_f_y, x0)

contour_with_path(f, x, y, paths, minima=np.array([2, 2]))

def contour_with_quiver(f, x, y, grad_x, grad_y, norm=LogNorm(), level = np.logspace(0, 5, 35),
    minima=None):
    dz_dx = grad_x(x,y)
    dz_dy = grad_y(x,y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
    if minima is not None:
        ax.plot(*minima, 'r*', makersize=18)
        ax.quiver(x, y, -dz_dx, -dz_dy, alpha=.5)
        ax.set_label('$x$')
        ax.set_ylabel('$y$')

        plt.show()

def surf(f, x, y, norm=LogNorm(), minima=None):
    fig = plt.figure(figsize=(8, 5))
def contour_with_quiver(f, x, y, grad_x, grad_y, norm=LogNorm(), level = np.logspace(0, 5, 35),
    minima=None):
    dz_dx = grad_x(x,y)
    dz_dy = grad_y(x,y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
    if minima is not None:
        ax.plot(*minima, 'r*', markersize=18)
        ax.quiver(x, y, -dz_dx, -dz_dy, alpha=.5)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        plt.show()

def surf(f, x, y, norm=LogNorm(), minima=None):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50, azim=-50)
    ax.plot_surface(x, y, f(x,y), norm=norm, rstride=1, cstride=1,
                    edgecolor='none', alpha=.8, cmap=plt.cm.jet)
    if minima is not None:
        ax.plot(*minima, f(*minima), 'r*', markersize=10)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    plt.show()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    plt.show()

def contour_with_path(f, x, y, paths, norm=LogNorm(), level = np.logspace(0, 5, 35),minima=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
    ax.quiver(paths[0:-1], paths[1:-1], paths[0,1:]-paths[0,:-1], paths[1,1:]-paths[1,:-1],
              scale_units='xy', angles='xy', scale=1, color='k')
    if minima is not None:
        ax.plot(*minima, 'r*', markersize=18)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

    plt.show()

import numpy as np
import matplotlib.pylab as plt
np.random.seed(320)
x_train = np.linspace(-1, 1, 51)
f = lambda x: 0.5 * x + 1.0
y_train = f(x_train) + 0.4 * np.random.rand(len(x_train))
plt.plot(x_train, y_train, 'o')
plt.grid()
plt.show()

np.random.seed(303)
shuffled_id = np.arange(0, len(x_train))
np.random.shuffle(shuffled_id)
x_train = x_train[shuffled_id]
y_train = y_train[shuffled_id]

def loss(w, x_set, y_set):
    N = len(x_set)
    val = 0.0
    for i in range(len(x_set)):
        val += (w[0] * x_set[i] - y_set[i]) ** 2
    return val / N

def loss_grad(w, x_set, y_set):
    N = len(x_set)
    val = np.zeros(len(w))
    for i in range(len(x_set))