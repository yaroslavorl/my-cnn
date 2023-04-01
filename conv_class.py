import numpy as np


class Conv2D(object):

    def __init__(self, filters, kernel_size, input_size):
        self.weights = np.random.uniform(0, 1, (kernel_size, kernel_size))
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_size = input_size

    def __call__(self, x):
        return self._conv(x, self.input_size, self.kernel_size)

    def _conv(self, x, input_size, kernel_size):
        output = np.zeros((input_size - kernel_size + 1, input_size - kernel_size + 1))

        for k in range(input_size - kernel_size + 1):
            for m in range(input_size - kernel_size + 1):
                output[k][m] = np.sum(x[k:kernel_size + k, m:kernel_size + m] * self.weights)

        return output
