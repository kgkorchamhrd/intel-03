
import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return x**2 - 4*x +6

Number0fPoints = 101
x= np.linspace(-5., 5, Number0fPoints)
fx= f(x)
plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.show()

xid = np.argmin(fx)
xopt =x[xid]
print(xopt, f(xopt))

plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.plot(xopt, f(xopt), 'xr')
plt.show()

def f(x):
  return x**2 - 4*x +6

def grad_fx(x):
  return 2*x -4

def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxlter=10, verbose=True):
  paths =[]
  for i in range(Maxlter):
    x1= x0- learning_rate* grad_func(x0)
    if verbose:
      print('{0:03d}:{1:4.3f}, {2:4.2E}'.format(i, x1, func(x1)))
    x0=x1
    paths.append(x0)
  return(x0, func(x0), paths)

xopt, fopt, paths = steepest_descent(f, grad_fx, 0.0, learning_rate=1.2)

def f(x):
  return x**2 -4*x +6

def grad_fx(x):
  return 2*x-4

x=np.linspace(-5, 5, 1000)
paths = np.array(paths)
plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths), '-o')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()

xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=1)

x=np.linspace(0.5,3.5,1000)
paths = np.array(paths)
plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths),'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()

xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=0.001)

x=np.linspace(0.5,3.5,1000)
paths = np.array(paths)
plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title ('plot of f(x)')
plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths))
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()

xop, fopt, paths = steepest_descent(f, grad_fx, 3.0, learning_rate=0.9)

x=np.linspace(0.5, 3.5, 1000)
paths = np.array(paths)
plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths))
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

x,y = np.meshgrid(np.arange(xmin, xmax+xstep, xstep),
                  np.arange(ymin, ymax+ystep, ystep))

f=lambda x,y : (x-2)**2 + (y-2)**2
z=f(x,y)
minima = np.array([2., 2.])
f(*minima)

minima_ = minima.reshape(-1,1)
print(minima,minima_)
surf(f,x,y, minima=minima_)

grad_f_x = lambda x, y: 2 * (x-2)
grad_f_y = lambda x, y: 2 * (y-2)

contour_with_quiver(f, x, y,grad_f_x, grad_f_y, minima=minima_)

def steepest_descent_twod(func, gradx, grady, x0, Maxiter=10,
                          learning_rate = 0.25, verbose=True):
  paths=[x0]
  fval_paths = [f(x0[0],x0[1])]
  for i in range(Maxiter):
    grad = np.array([grad_f_x(*x0),grad_f_y(*x0)])
    x1= x0 - learning_rate *grad
    fval =f(*x1)
    if verbose:
      print(i, x1, fval)
    x0=x1
    paths.append(x0)
    fval_paths.append(fval)
  paths = np.array(paths)
  paths = np.array(np.matrix(paths).T)
  fval_pahts = np.array(fval_paths)
  return(x0,fval, paths, fval_paths)

x0 = np.array([-2., -2.])
xopt, fopt, paths, fval_paths = steepest_descent_twod(f, grad_f_x, grad_f_y, x0)

contour_with_path(f,x,y,paths, minima=np.array([[2],[2]]))