import numpy as np
import matplotlib.pyplot as plt

np.random.seed(320)
x_train = np.linspace(-1, 1, 51)
f = lambda x:0.5 * x +1.0
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
    N=len(x_set)
    val=0.0
    for i in range(len(x_set)):
      val +=0.5 * (w[0] * x_set[i] + w[1] -y_set[i] )**2
    return val /N
def loss_grad(w, x_set, y_set):
    N =  len(x_set)
    val = np.zeros(len(w))
    for i in range(len(x_set)):
      er = w[0] * x_set[i] + w[1] -y_set[i]
      val += er * np.array([x_set[i], 1.0])
    return val /N

def generate_batches(batch_size, features, labels):
  assert len(features) == len(labels)
  outout_batches =[]

  sample_size = len(features)
  for start_i in range(0,sample_size, batch_size):
    end_i = start_i + batch_size
    batch = [features[start_i:end_i], labels[start_i:end_i]]
    outout_batches.append(batch)
  return outout_batches

batch_size = 10
lr = 0.01
MaxEpochs =51

alpha = .9

w0 = np.array([4.0,-1.0])
path_sgd = []
for epoch in range(MaxEpochs):
    if epoch %10 ==0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size,
                                             x_train, y_train):
        path_sgd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        w1 = w0 - lr * grad
        w0 =w1

w0 = np.array([4.0, -1.0])
path_mm = []
velocity =np.zeros_like(w0)
for epoch in range(MaxEpochs):
  if epoch %10 ==0:
    print(epoch, w0, loss(w0, x_train, y_train))
  for x_batch, y_batch in generate_batches(batch_size,
                                           x_train, y_train):
    path_mm.append(w0)
    grad = loss_grad(w0, x_batch, y_batch)
    velocity = alpha * velocity -lr * grad
    w1 = w0 + velocity
    w0 = w1

# Assuming loss function and x_train, y_train are defined elsewhere

W0 = np.linspace(-2, 5, 101)
W1 = np.linspace(-2, 5, 101)
W0, W1 = np.meshgrid(W0, W1)
LOSSW = W0 * 0  # Initialize LOSSW with zeros

for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        wij = np.array([W0[i, j], W1[i, j]])
        LOSSW[i, j] = loss(wij, x_train, y_train)  # Replace 'loss' with your actual loss function

fig, ax = plt.subplots(figsize=(6, 6))

ax.contour(W0, W1, LOSSW, cmap=plt.cm.jet,
           levels=np.linspace(0, max(LOSSW.flatten()), 20))

paths = path_sgd  # Assuming path_sgd is defined elsewhere

paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] -paths[0, :-1],
          paths[1, 1:] -paths[1, :-1],
          scale_units='xy', angles='xy', scale=1, color='k')

paths = path_mm # Assuming path_mm is defined elsewhere

paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0,1:] - paths[0, :-1],
          paths[1, 1:] -paths[1, :-1],
          scale_units='xy', angles='xy', scale=1, color='r')

plt.legend(['GD', 'Moment'])
plt.show()

# SGD
batch_size = 10
lr = 1.5
MaxEpochs = 51

# Adagrad
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
        delw = - epsilon / (delta + np.sqrt(r)) * grad
        w1 = w0 + delw
        w0 = w1

# Assuming W0, W1, LOSSW are defined elsewhere

fig, ax = plt.subplots(figsize=(6, 6))

ax.contour(W0, W1, LOSSW, cmap=plt.cm.jet,
           levels=np.linspace(0, max(LOSSW.flatten()), 20))

paths = path_sgd

paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1],
          paths[1, 1:] - paths[1, :-1], scale_units='xy',
          angles='xy', scale=1, color='k')  # Fixed typo here

paths = path_adagd

paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1],
          paths[1, 1:] - paths[1, :-1], scale_units='xy',
          angles='xy', scale=1, color='r') # Fixed typo here

plt.legend(['GD', 'Adagrad'])
plt.show()

MaxEpochs = 51
batch_size = 10

# Adagrad
epsilon = 0.25
delta = 1E-6

# RMSProp
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
        delw = - epsilon / (delta + np.sqrt(r)) * grad
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
        delw = - epsilon * grad / np.sqrt(delta + r)
        w1 = w0 + delw
        w0 = w1

import matplotlib.pyplot as plt
import numpy as np

# Assuming W0, W1, LOSSW are defined elsewhere

fig, ax = plt.subplots(figsize=(6, 6))

ax.contour(W0, W1, LOSSW, cmap=plt.cm.jet,
           levels=np.linspace(0, max(LOSSW.flatten()), 20))

paths = path_adagd

paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1],
          paths[1, 1:] - paths[1, :-1], scale_units='xy',
          angles='xy', scale=1, color='k')

paths = path_rmsprop

paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1],
          paths[1, 1:] - paths[1, :-1], scale_units='xy',
          angles='xy', scale=1, color='r')

plt.legend(['Adagrad', 'RMSProp'])
plt.show()

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

for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))

    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_rmsprop.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        r = rho * r + (1. - rho) * grad * grad
        delw = - epsilon * grad / np.sqrt(delta + r)
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
        shat = s / (1. - rho1 ** t)
        rhat = r / (1. - rho2 ** t)
        delw = - epsilon * shat / (delta_adam + np.sqrt(rhat))
        w1 = w0 + delw
        w0 = w1

import matplotlib.pyplot as plt
import numpy as np

# Assuming W0, W1, LOSSW are defined elsewhere

fig, ax = plt.subplots(figsize=(6, 6))

ax.contour(W0, W1, LOSSW, cmap=plt.cm.jet,
           levels=np.linspace(0, max(LOSSW.flatten()), 20))

paths = path_rmsprop

paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1],
          paths[1, 1:] - paths[1, :-1], scale_units='xy',
          angles='xy', scale=1, color='k')

paths = path_adam

paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1],
          paths[1, 1:] - paths[1, :-1], scale_units='xy',
          angles='xy', scale=1, color='r')

plt.legend(['RMSProp', 'Adam'])
plt.show()
