import numpy as np
import matplotlib.pylab as plt

np.random.seed(320)
x_train = np.linspace(-1, 1, 51)
f = lambda x:0.5 * x +1.0
y_train = f(x_train) + 0.4 * np.random.rand(len(x_train))
plt.plot(x_train, y_train, 'o')
plt.grid()
plt.show()

