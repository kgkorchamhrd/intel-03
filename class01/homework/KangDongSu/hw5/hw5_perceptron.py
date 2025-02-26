import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # np.exp(-x)로 수정


def numerical_derivative(f, x):
    dx = 1e-4
    gradf = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + dx
        fx1 = f(x)

        x[idx] = tmp_val - dx
        fx2 = f(x)

        gradf[idx] = (fx1 - fx2) / (2 * dx)
        x[idx] = tmp_val  # 원래 값 복구

        it.iternext()

    return gradf


class LogicGate:
    def __init__(self, gate_name, xdata, tdata, learning_rate=0.01, threshold=0.5):
        self.name = gate_name
        self.xdata = xdata.reshape(4, 2)
        self.tdata = tdata.reshape(4, 1)

        self.w = np.random.rand(2, 1)
        self.b = np.random.rand(1)
        self.learning_rate = learning_rate
        self.threshold = threshold

    def _loss_func(self):
        delta = 1e-7
        z = np.dot(self.xdata, self.w) + self.b
        y = sigmoid(z)

        return -np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

    def err_val(self):
        delta = 1e-7
        z = np.dot(self.xdata, self.w) + self.b
        y = sigmoid(z)

        return -np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

    def train(self):
        f = lambda _: self._loss_func()
        print("Init error:", self.err_val())

        for step in range(20000):
            grad_w = numerical_derivative(f, self.w)
            grad_b = numerical_derivative(f, self.b)

            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

            if step % 2000 == 0:
                print("Step:", step, "Error:", self.err_val())

    def predict(self, input_data):
        z = np.dot(input_data, self.w) + self.b
        y = sigmoid(z)

        result = (y > self.threshold).astype(int)
        return y, result


xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0], [0], [0], [1]])  # (4,1) 형태로 수정
AND = LogicGate("AND", xdata, tdata)
AND.train()
for in_data in xdata:
    sig_val, logic_val = AND.predict(in_data)
    print(in_data, ":", logic_val[0])

xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0], [1], [1], [1]])  # (4,1) 형태로 수정
OR = LogicGate("OR", xdata, tdata)
OR.train()
for in_data in xdata:
    sig_val, logic_val = OR.predict(in_data)
    print(in_data, ":", logic_val[0])
