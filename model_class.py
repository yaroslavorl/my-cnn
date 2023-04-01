from class_cnn.conv_class import Conv2D
import numpy as np


class ModelCnn(object):

    def __init__(self):
        self.layers_1 = Conv2D(1, kernel_size=2, input_size=5)

    def predict(self, x):
        return self._relu(self.layers_1(x))

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)


if __name__ == "__main__":
    img = np.random.randint(0, 256, (1, 5, 5))
    model = ModelCnn()
    print(model.predict(img).shape)

