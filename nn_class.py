import numpy as np


class Conv2D(object):

    def __init__(self, filters, kernel_size, input_size):
        self.weights = np.random.uniform(0, 1, (filters, kernel_size, kernel_size))
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_size = input_size

    def __call__(self, x):
        return self._conv(x, self.input_size, self.kernel_size)

    def _conv(self, x, input_size, kernel_size):
        output = np.zeros((self.filters, input_size - kernel_size + 1, input_size - kernel_size + 1))

        for k in range(input_size - kernel_size + 1):
            for m in range(input_size - kernel_size + 1):
                output[:, k, m] = np.sum(x[0, k:kernel_size + k, m:kernel_size + m] * self.weights, axis=(1, 2))

        return output


class Poling(object):
    _METHOD = {'max': np.max, 'min': np.min, 'avg': np.average}

    def __init__(self, input_size, input_channels, pool_size, method='max'):
        self.pool_size = pool_size
        self.input_size = input_size
        self.input_channels = input_channels
        self.method = self._METHOD[method]

    def __call__(self, x):
        return self.poling(x)

    def poling(self, x):
        output_size = self.input_size // self.pool_size
        output = np.zeros((self.input_channels, output_size, output_size))

        for channel in range(self.input_channels):
            for i, k in zip(range(0, self.input_size, self.pool_size), range(output_size)):
                for j, m in zip(range(0, self.input_size, self.pool_size), range(output_size)):
                    output[channel, k, m] = self.method(x[channel, i:self.pool_size + i, j:self.pool_size + j])
        return output


class BatchNormalization(object):

    def __init__(self, eps=1e5):
        self.weights = np.array([1, 0])
        self.eps = eps

    def __call__(self, x):
        return self.batch_norm(x)

    def batch_norm(self, x):

        m = np.mean(x)
        disp = np.sum((x - m) ** 2) / m
        x_new = (x - m)/(np.sqrt(disp + self.eps))

        return x_new * self.weights[0] + self.weights[1]
