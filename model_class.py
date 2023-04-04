from class_cnn.nn_class import Conv2D, Poling
import numpy as np


class ModelCnn(object):

    def __init__(self):
        self.layers_1 = Conv2D(2, kernel_size=3, input_size=32)
        self.pool = Poling(30, 2, pool_size=2, method='min')

    def predict(self, x):
        x = self.layers_1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)


if __name__ == "__main__":
    img = np.random.randint(0, 255, (1, 32, 32))
    model = ModelCnn()
    print(model.predict(img).shape)

