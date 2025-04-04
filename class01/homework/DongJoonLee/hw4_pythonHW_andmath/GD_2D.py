import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from visualize import contour_with_quiver
from visualize import contour_with_path
from visualize import surf

xmin, xmax, xstep = -4.0, 4.0, .25
ymin, ymax, ystep = -4.0, 4.0, .25

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                   np.arange(ymin, ymax + ystep, ystep))

f = lambda x, y : (x-2)**2 + (y-2)**2
z = f(x,y)
minima = np.array([2., 2.])

f(*minima)

minima_ = minima.reshape(-1,1)
print(minima, minima_)
surf(f, x, y, minima=minima_)

grad_f_x = lambda x, y: 2 * (x-2)
grad_f_y = lambda x, y: 2 * (y-2)

contour_with_quiver(f, x, y, grad_f_x, grad_f_y, minima=minima_)

def steepest_descent_twod(func, gradx, grady, x0, Maxlter=10, learning_rate=0.25, verbose=True):
    paths = [x0]
    fval_paths = [func(x0[0], x0[1])]
    for i in range(Maxlter):
        grad = np.array([gradx(*x0), grady(*x0)])
        x1 = x0 - learning_rate * grad
        fval = func(*x1)
        if verbose:
            print(i, x1, fval)
        x0 = x1
        paths.append(x0)
        fval_paths.append(fval)
    paths = np.array(paths)
    paths = np.array(np.matrix(paths).T)
    fval_paths = np.array(fval_paths)
    return (x0, fval, paths, fval_paths)

x0= np.array([-2., -2.])
xopt, fopt, paths, fval_paths = steepest_descent_twod(f, grad_f_x, grad_f_y, x0)

contour_with_path(f, x, y, paths, minima=np.array([[2],[2]]))

