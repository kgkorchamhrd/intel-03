# ==================================================================================================
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')


def f(x):
    return x**2 - 4*x + 6


NumberOfPoints = 101

x = np.linspace(-5, 5, NumberOfPoints)
fx = f(x)


plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.show()


# ==================================================================================================
xid = np.argmin(fx)
xopt = x[xid]

plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(xopt, f(xopt), 'xr')
plt.show()


# ==================================================================================================
def f(x):
    return x**2 - 4*x + 6


def grad_fx(x):
    return 2*x - 4

# ==================================================================================================
def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxlter=10, verbose=True):
    paths = []

    for i in range(Maxlter):
        x1 = x0 - learning_rate * grad_func(x0)

        if verbose:
            print('{0:03d} : {1:4.3f}, {2:4.2E}'.format(i, x1, func(x1)))
        x0 = x1
        paths.append(x0)
    return (x0, func(x0), paths)


xopt, fopt, paths = steepest_descent(f, grad_fx, 0.0, learning_rate=1.2)

x = np.linspace(0.5, 2.5, 1000)
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


# ==================================================================================================
xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=1)

x = np.linspace(0.5, 3.5, 1000)
paths = np.array(paths)
plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.show()

plt.plot(f(paths), 'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()



# ==================================================================================================
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

plt.plot(f(paths))
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()



# ==================================================================================================
xopt, fopt, paths = steepest_descent(f, grad_fx, 3.0, learning_rate=0.9)

x = np.linspace(0.5, 3.5, 1000)
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


# ==================================================================================================
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

def contour(f, x, y, level = np.logspace(0, 5, 35)):
	fig, ax = plt.subplots(figsize=(8, 8))

	ax.contour(x, y, f(x,y), levels=level, norm=LogNorm(), cmap=plt.cm.jet)

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()

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

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

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

def contour_with_path(f, x, y, paths, norm=LogNorm(), level=np.logspace(0, 5, 35), minima=None):
	fig, ax = plt.subplots(figsize=(6, 6))

	ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
	ax.quiver(paths[0,:-1], 
		      paths[1,:-1], 
			  paths[0,1:]-paths[0,:-1], 
			  paths[1,1:]-paths[1,:-1], 
			  scale_units='xy', 
			  angles='xy', 
			  scale=1, 
			  color='k')
	
	if minima is not None:
		ax.plot(*minima, 'r*', markersize=18)

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()



# ==================================================================================================
xmin, xmax, xstep = -4.0, 4.0, .25
ymin, ymax, ystep = -4.0, 4.0, .25

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
				   np.arange(ymin, ymax + ystep, ystep))

f = lambda x,y : (x-2)**2 + (y-2)**2
z = f(x, y)
minima = np.array([2., 2.])

f(*minima)

minima_ = minima.reshape(-1, 1)
print(minima, minima_)
surf(f, x, y, minima=minima_)

grad_f_x =lambda x, y: 2 * (x-2)
grad_f_y =lambda x, y: 2 * (y-2)

contour_with_quiver(f, x, y, grad_f_x, grad_f_y, minima=minima_)


def steepest_descent_twod(func, gradx, grady, x0, Maxlter=10, learning_rate=0.25, verbose=True):
    paths = [x0]
    fval_paths = [f(x0[0], x0[1])]
    
    for i in range(Maxlter):
        grad = np.array([grad_f_x(*x0), grad_f_y(*x0)])
        x1 = x0 - learning_rate * grad
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

x0 = np.array([-2, -2,])
xopt, fopt, paths, fval_paths = steepest_descent_twod(f, grad_f_x, grad_f_y, x0)

contour_with_path(f, x, y, paths, minima=np.array([[2], [2]]))



# ==================================================================================================
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
shuffle_id = np.arange(0, len(x_train))
np.random.shuffle(shuffle_id)
x_train = x_train[shuffle_id]
y_train = y_train[shuffle_id]



# ==================================================================================================
def loss(w, x_set, y_set):
    N = len(x_set)
    val = 0.0
    for i in range(len(x_set)):
        val += 0.5 * ( w[0] * x_set[i] + w[1] - y_set[i] )**2
    return val/N

def loss_grad(w, x_set, y_set):
    N = len(x_set)
    val = np.zeros(len(w))
    for i in range(len(x_set)):
        er = w[0] * x_set[i] + w[1] - y_set[i]
        val += er * np.array([x_set[i], 1.0])
    return val/N

def generate_batches(batch_size, features, labels):
    assert len(features) == len(labels)
    output_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
    
    return output_batches



# ==================================================================================================
batch_size = 10
lr = 0.01
MaxEpochs = 51

alpha = .9

w0 = np.array([4.0, -1.0])
path_sgd = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_sgd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        w1 = w0 - lr * grad
        w0 = w1

w0 = np.array([4.0, -1.0])
path_mm = []
velocity = np.zeros_like(w0)
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_mm.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        velocity = alpha * velocity - lr * grad
        w1 = w0 + velocity
        w0 = w1

W0 = np.linspace(-2, 5, 101)
W1 = np.linspace(-2, 5, 101)

W0, W1 = np.meshgrid(W0, W1)

LOSSW = W0 * 0
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        wij = np.array([W0[i,j], W1[i,j]])
        LOSSW[i, j] = loss(wij, x_train, y_train)

fig, ax = plt.subplots(figsize=(6,6))

ax.contour(W0, W1, LOSSW, cmap=plt.cm.jet, levels=np.linspace(0, max(LOSSW.flatten()),20))

paths = path_sgd
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], 
          paths[1, :-1], 
          paths[0, 1:] - paths[0, :-1], 
          paths[1, 1:] - paths[1, :-1],
          scale_units='xy',
          angles='xy',
          scale=1,
          color='k')

paths = path_mm
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], 
          paths[1, :-1], 
          paths[0, 1:] - paths[0, :-1],
          paths[1, 1:] - paths[1, :-1],
          scale_units='xy',
          angles='xy',
          scale=1,
          color='r')

plt.legend(['GD', 'Moment'])
plt.show()



# ==================================================================================================
batch_size = 10
lr = 1.5
MaxEpochs = 51

epsilon = lr
delta = 1E-7

w0 = np.array([4.0, -1.0])
path_sgd = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_sgd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        w1 = w0 - lr * grad
        w0 = w1


w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)
path_adagd = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_adagd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        r = r + grad * grad
        delw = -epsilon / (delta + np.sqrt(r)) * grad
        w1 = w0 + delw
        w0 = w1

fig, ax = plt.subplots(figsize=(6, 6))

ax.contour(W0, W1, LOSSW, cmap=plt.cm.jet, levels=np.linspace(0, max(LOSSW.flatten()),20))
paths = path_sgd
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1],
          paths[1, :-1],
          paths[0, 1:]-paths[0, :-1],
          paths[1, 1:]-paths[1, :-1], 
          scale_units='xy',
          angles='xy',
          scale=1,
          color='k')

paths = path_adagd
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1],
          paths[1, :-1],
          paths[0, 1:]-paths[0, :-1],
          paths[1, 1:]-paths[1, :-1],
          scale_units='xy',
          angles='xy',
          scale=1,
          color='r')

plt.legend(['GD', 'Adagrad'])
plt.show()



# ==================================================================================================
MaxEpochs = 51
batch_size = 10

epsilon = 0.25
delta = 1E-6

rho = 0.9

w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)
path_adagd = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_adagd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        r = r + grad * grad
        delw = -epsilon / (delta + np.sqrt(r)) * grad
        w1 = w0 + delw
        w0 = w1

w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)
path_rmsprop = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_rmsprop.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        r = rho * r + (1. - rho) * grad * grad
        delw = -epsilon *grad / np.sqrt(delta + r)
        w1 = w0 + delw
        w0 = w1


fig, ax = plt.subplots(figsize=(6, 6))

ax.contour(W0,W1, LOSSW, cmap=plt.cm.jet, levels=np.linspace(0, max(LOSSW.flatten()),20))
paths = path_adagd
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1],
          paths[1, :-1],
          paths[0, 1:]-paths[0,:-1],
          paths[1, 1:]-paths[1, :-1],
          scale_units='xy',
          angles='xy',
          scale=1,
          color='k')

paths = path_rmsprop
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1],
          paths[1, :-1],
          paths[0, 1:]-paths[0,:-1],
          paths[1, 1:]-paths[1, :-1],
          scale_units='xy',
          angles='xy',
          scale=1,
          color='r')

plt.legend(['Adagrad', 'RMSProp'])
plt.show()



# ==================================================================================================
MaxEpochs = 51
batch_size = 10
epsilon = 0.1
delta = 1E-6

rho = 0.9

delta_adam = 1E-8
rho1 = 0.9
rho2 = 0.999

w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)
path_rmsprop = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_rmsprop.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        r = rho * r + (1. - rho) * grad * grad
        delw = -epsilon * grad / np.sqrt(delta + r)
        w1 = w0 + delw
        w0 = w1


w0 = np.array([4.0, -1.0])
s = np.zeros_like(w0)
r = np.zeros_like(w0)

path_adam = []
t = 0

for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_adam.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        s = rho1 * s + (1. - rho1) * grad
        r = rho2 * r + (1. - rho2) * (grad * grad)
        t += 1
        shat = s/(1. - rho1 ** t)
        rhat = r/(1. - rho2 ** t)
        delw = -epsilon * shat/(delta_adam + np.sqrt(rhat))
        w1 = w0 + delw
        w0 = w1

fig, ax = plt.subplots(figsize=(6,6))

ax.contour(W0, W1, LOSSW, cmap=plt.cm.jet, levels=np.linspace(0, max(LOSSW.flatten()),20))

paths = path_rmsprop
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1],
          paths[1, :-1],
          paths[0, 1:]-paths[0,:-1],
          paths[1, 1:]-paths[1, :-1],
          scale_units='xy',
          angles='xy',
          scale=1,
          color='k')

paths = path_adam
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1],
          paths[1, :-1],
          paths[0, 1:]-paths[0,:-1],
          paths[1, 1:]-paths[1, :-1],
          scale_units='xy',
          angles='xy',
          scale=1,
          color='r')

plt.legend(['RMSProp', 'Adam'])
plt.show()