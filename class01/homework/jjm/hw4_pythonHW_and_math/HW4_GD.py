
#HW4_GD.py


import numpy as np
import matplotlib.pyplot as plt

# exam01
def f(x):
    return x ** 2 - 4 * x + 6

NumberOfPoints = 101
x = np.linspace(-5 , 5, NumberOfPoints)
fx = f(x)
print(x)
print(fx)
plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('fx')
plt.title('plot of f(x)')
plt.savefig('./src_img/01_x_fx.png', dpi=300, bbox_inches='tight')
plt.close()


# exam02
xid = np.argmin(fx)
xopt = x[xid]
print(xopt, f(xopt))

plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('fx')
plt.title('plot of f(x)')
plt.plot(xopt, f(xopt), 'xr')
plt.savefig('./src_img/02_xopt_fxopt.png', dpi=300, bbox_inches='tight')
plt.close()


# exam03
def gred_f(x):
    return 2 * x - 4

def steepest_descent(func, grad_func, x0, learning_rate = 0.01, Maxlter = 10, verbose = True):
    paths = []
    for i in range(Maxlter):
        x1 = x0 - learning_rate * grad_func(x0)

        if verbose:
            print('{0:03d} : {1:4.3f}. {2:4.2E}'.format(i, x1, func(x1)))
        x0 = x1
        paths.append(x0)
    return (x0, func(x0), paths)

xopt, fopt, paths = steepest_descent(f, gred_f, 0.0, learning_rate = 1.2)

x = np.linspace(0.5, 2.5, 1000)
fx = f(x)
paths = np.array(paths)
plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('fx')
plt.title('plot of f(x)')

plt.plot(paths, f(paths), '-o')
plt.savefig('./src_img/03_0-LR-1d2.png', dpi=300, bbox_inches='tight')
plt.close()


# exam04
plt.plot(f(paths), '-o')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.savefig('./src_img/04_0-LR-1d2_cost.png', dpi=300, bbox_inches='tight')
plt.close()


# exam05
xopt, fopt, paths = steepest_descent(f, gred_f, 1.0, learning_rate = 1)
x = np.linspace(0.5, 3.5, 1000)
fx = f(x)
paths = np.array(paths)
plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('fx')
plt.title('plot of f(x)')
plt.plot(paths, f(paths), 'o-')
plt.savefig('./src_img/05_1-LR-1.png', dpi=300, bbox_inches='tight')
plt.close()

# plt.plot(f(paths), 'o-')
plt.plot(f(paths))
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.savefig('./src_img/06_1-LR-1_cost.png', dpi=300, bbox_inches='tight')
plt.close()


# exam06
xopt, fopt, paths = steepest_descent(f, gred_f, 1.0, learning_rate = 0.001)
x = np.linspace(0.5, 3.5, 1000)
fx = f(x)
paths = np.array(paths)
plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('fx')
plt.title('plot of f(x)')
plt.plot(paths, f(paths), 'o-')
plt.savefig('./src_img/07_1-LR-0d001.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(f(paths))
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.savefig('./src_img/08_1-LR-0d001_cost.png', dpi=300, bbox_inches='tight')
plt.close()


# exam07
xopt, fopt, paths = steepest_descent(f, gred_f, 3.0, learning_rate = 0.9)
x = np.linspace(0.5, 3.5, 1000)
fx = f(x)
paths = np.array(paths)
plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('fx')
plt.title('plot of f(x)')
plt.plot(paths, f(paths), 'o-')
plt.savefig('./src_img/09_3-LR-0d9.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(f(paths))
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.savefig('./src_img/10_3-LR-0d9_cost.png', dpi=300, bbox_inches='tight')
plt.close()


# exam08
from matplotlib.colors import LogNorm
from visualize import contour_with_quiver
from visualize import contour_with_path
from visualize import surf

xmin, xmax, xstep = -4.0, 4.0, .25
ymin, ymax, ystep = -4.0, 4.0, .25

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                   np.arange(ymin, ymax + ystep, ystep))


f = lambda x, y : (x - 2) ** 2 + (y - 2) ** 2
z = f(x, y)
minima = np.array([2., 2.])

f(*minima)

minima_ = minima.reshape(-1, 1)
print(minima, minima_)
img_path = './src_img/11_2D_GradientDescent_01.png'
surf(f, x, y, minima = minima_, img_path = img_path)

grad_f_x = lambda x, y : 2 * (x - 2)
grad_f_y = lambda x, y : 2 * (y - 2)

img_path = './src_img/12_2D_GradientDescent_02.png'
contour_with_quiver(f, x, y, grad_f_x, grad_f_y, minima = minima_, img_path = img_path)
plt.close()


def steepest_descent_twod(func, gradx, grady, x0, Maxlter = 10, learning_rate = 0.25, verbose = True):
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

x0 = np.array([-2., -2.])
xopt, fopt, paths, fval_paths = steepest_descent_twod(f, grad_f_x, grad_f_y, x0)
img_path = './src_img/13_2D_GradientDescent_03.png'
contour_with_path(f, x, y, paths, minima = np.array([[2], [2]]), img_path = img_path)



# exam09
np.random.seed(320)
x_train = np.linspace(-1, 1, 51)
f = lambda x : 0.5 * x  + 1.0
y_train = f(x_train) + 0.4 * np.random.rand(len(x_train))
plt.plot(x_train, y_train, 'o')
plt.grid()
plt.savefig('./src_img/14_GD_vs_Momentum_Data.png', dpi=300, bbox_inches='tight')
plt.close()

np.random.seed(303)
shuffled_id = np.arange(0, len(x_train))
np.random.shuffle(shuffled_id)
x_train = x_train[shuffled_id]
y_train = y_train[shuffled_id]


def loss(w, x_set, y_set):
    N = len(x_set)
    val = 0.0
    for i in range(len(x_set)):
        val += 0.5 * (w[0] * x_set[i] + w[1] - y_set[i]) ** 2
    return val / N

def loss_grad(w, x_set, y_set):
    N = len(x_set)
    val = np.zeros(len(w))  
    for i in range(len(x_set)):
        er = w[0] * x_set[i] + w[1] - y_set[i]
        val += er * np.array([x_set[i], 1.0])
    return val / N

def generate_batches(batch_size, features, labels):
    """
    Create batchs of features and labels
    :param batch_size: the batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Feature, Labels)
    """
    assert len(features) == len(labels)
    outout_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i : end_i], labels[start_i : end_i]]
        outout_batches.append(batch)

    return outout_batches 

# SGD
batch_size = 10
lr = 0.01
MaxEpochs = 51

# Momentum
alpha = .9

# SGD
w0 = np.array([4.0, -1.0])
path_sgd = []
for epochs in range(MaxEpochs):
    if epochs % 10 == 0:
        print(epochs, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_sgd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        w1 = w0 - lr * grad
        w0 = w1

# Momentum
w0 = np.array([4.0, -1.0])
path_mm = []
velocity = np.zeros_like(w0)
for epochs in range(MaxEpochs):
    if epochs % 10 == 0:
        print(epochs, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_mm.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        velocity = alpha * velocity - lr * grad
        w1 = w0 - lr * grad
        w0 = w1


W0 = np.linspace(-2, 5, 101)
W1 = np.linspace(-2, 5, 101)

W0, W1 = np.meshgrid(W0, W1)
LOSSW = W0 * 0
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        wij = np.array([W0[i, j], W1[i, j]])
        LOSSW[i, j] = loss(wij, x_train, y_train)

fig, ax = plt.subplots(figsize = (6, 6))

ax.contour(W0, W1, LOSSW, cmap = plt.cm.jet, levels = np.linspace(0, max(LOSSW.flatten()), 20))
paths = path_sgd
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, : -1], paths[1, : -1], paths[0, 1 :] - paths[0, : -1], paths[1, 1 :] - paths[1, : -1],
          scale_units = 'xy', angles = 'xy', scale = 1, color = 'k')

paths = path_mm
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, : -1], paths[1, : -1], paths[0, 1 :] - paths[0, : -1], paths[1, 1 :] - paths[1, : -1],
          scale_units = 'xy', angles = 'xy', scale = 1, color = 'r')

plt.legend(['GD', 'Momemtum'])
plt.savefig('./src_img/15_GD_vs_Momentum.png', dpi=300, bbox_inches='tight')
plt.close()


# exam10
def loss(w, x_set, y_set):
    N = len(x_set)
    val = 0.0
    for i in range(len(x_set)):
        val += 0.5 * (w[0] *x_set[i] + w[1] - y_set[i]) ** 2
    return val / N

def loss_grad(w, x_set, y_set):
    N = len(x_set)
    val = np.zeros(len(w))  
    for i in range(len(x_set)):
        er = w[0] * x_set[i] + w[1] - y_set[i]
        val += er * np.array([x_set[i], 1.0])
    return val / N

def generate_batches(batch_size, features, labels):
    """
    Create batchs of features and labels
    :param batch_size: the batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Feature, Labels)
    """
    assert len(features) == len(labels)
    outout_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i : end_i], labels[start_i : end_i]]
        outout_batches.append(batch)

    return outout_batches 



# SGD
batch_size = 10
lr = 0.01
MaxEpochs = 51

# Adadrad
epsilon = lr
delta = 1E-7

w0 = np.array([4.0, -1.0])
path_sgd = []
for epochs in range(MaxEpochs):
    if epochs % 10 == 0:
        print(epochs, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_sgd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        w1 = w0 - lr * grad
        w0 = w1

w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)
path_adadrad = []
for epochs in range(MaxEpochs):
    if epochs % 10 == 0:
        print(epochs, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_adadrad.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        r = r + grad * grad
        delw = -epsilon / (delta + np.sqrt(r)) * grad
        w1 = w0 + delw 
        w0 = w1


# W0 = np.linspace(-2, 5, 101)
# W1 = np.linspace(-2, 5, 101)

# W0, W1 = np.meshgrid(W0, W1)
# LOSSW = W0 * 0
# for i in range(W0.shape[0]):
#     for j in range(W0.shape[1]):
#         wij = np.array([W0[i, j], W1[i, j]])
#         LOSSW[i, j] = loss(wij, x_train, y_train)

fig, ax = plt.subplots(figsize = (6, 6))

ax.contour(W0, W1, LOSSW, cmap = plt.cm.jet, levels = np.linspace(0, max(LOSSW.flatten()), 20))
paths = path_sgd
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, : -1], paths[1, : -1], paths[0, 1 :] - paths[0, : -1], paths[1, 1 :] - paths[1, : -1],
          scale_units = 'xy', angles = 'xy', scale = 1, color = 'k')

paths = path_adadrad
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, : -1], paths[1, : -1], paths[0, 1 :] - paths[0, : -1], paths[1, 1 :] - paths[1, : -1],
          scale_units = 'xy', angles = 'xy', scale = 1, color = 'r')

plt.legend(['GD', 'Adadrad'])
plt.savefig('./src_img/16_GD_vs_Adadrad.png', dpi=300, bbox_inches='tight')
plt.close()


MaxEpochs = 51
batch_size = 10
# Adadrad
epsilon = 0.25
delta = 1E-6

#RMSProp
rho = 0.9

w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)
path_adadrad = []
for epochs in range(MaxEpochs):
    if epochs % 10 == 0:
        print(epochs, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_adadrad.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        r = r + grad * grad
        delw = -epsilon / (delta + np.sqrt(r)) * grad
        w1 = w0 + delw 
        w0 = w1



w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)
path_rmsprop = []
for epochs in range(MaxEpochs):
    if epochs % 10 == 0:
        print(epochs, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_rmsprop.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        r = rho * r + (1, - rho) * grad * grad
        delw = -epsilon * grad / np.sqrt(delta + r)
        w1 = w0 + delw 
        w0 = w1

fig, ax = plt.subplots(figsize = (6, 6))

ax.contour(W0, W1, LOSSW, cmap = plt.cm.jet, levels = np.linspace(0, max(LOSSW.flatten()), 20))
paths = path_adadrad
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, : -1], paths[1, : -1], paths[0, 1 :] - paths[0, : -1], paths[1, 1 :] - paths[1, : -1],
          scale_units = 'xy', angles = 'xy', scale = 1, color = 'k')

paths = path_rmsprop
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, : -1], paths[1, : -1], paths[0, 1 :] - paths[0, : -1], paths[1, 1 :] - paths[1, : -1],
          scale_units = 'xy', angles = 'xy', scale = 1, color = 'r')

plt.legend(['Adadrad', 'rmsprop'])
plt.savefig('./src_img/17_Adadrad_vs_rmsprop.png', dpi=300, bbox_inches='tight')
plt.close()



MaxEpochs = 51
batch_size = 10
epsilon = 0.1
delta = 1E-6

# RMSProp
rho = 0.9

# Adam
delta_adam = 1E-8
rho1 = 0.9
rho2 = 0.999

w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)
path_rmsprop = []
for epochs in range(MaxEpochs):
    if epochs % 10 == 0:
        print(epochs, w0, loss(w0, x_train, y_train))
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
for epochs in range(MaxEpochs):
    if epochs % 10 == 0:
        print(epochs, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_adam.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        s = rho1 * s + (1. - rho1) * grad
        r = rho2 * r + (1. - rho2) * (grad * grad)
        t += 1
        shat = s / (1. - rho1 ** t)
        rhat = r / (1. - rho2 ** t)
        delw = -epsilon * shat / (delta_adam + np.sqrt(rhat))
        w1 = w0 + delw 
        w0 = w1


fig, ax = plt.subplots(figsize = (6, 6))

ax.contour(W0, W1, LOSSW, cmap = plt.cm.jet, levels = np.linspace(0, max(LOSSW.flatten()), 20))
paths = path_rmsprop
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, : -1], paths[1, : -1], paths[0, 1 :] - paths[0, : -1], paths[1, 1 :] - paths[1, : -1],
          scale_units = 'xy', angles = 'xy', scale = 1, color = 'k')

paths = path_adam
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, : -1], paths[1, : -1], paths[0, 1 :] - paths[0, : -1], paths[1, 1 :] - paths[1, : -1],
          scale_units = 'xy', angles = 'xy', scale = 1, color = 'r')

plt.legend(['rmsprop', 'adam'])
plt.savefig('./src_img/18_rmsprop_vs_adam.png', dpi=300, bbox_inches='tight')
plt.close()

