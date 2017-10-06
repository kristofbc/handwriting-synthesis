import chainer
import chainer.functions as F
from chainer import variable

from links.connection.linear import Linear

class LinearLSTM(chainer.Chain):
    """
        Alex Graves' LSTM implementation
        Args:
            n_units (int): Number of units inside this LSTM
    """
    def __init__(self, n_units):
        super(LinearLSTM, self).__init__()

        self.n_units = n_units
        self.h = None
        self.c = None

        with self.init_scope():
            self.h_x = Linear(None, n_units)

    def reset_state(self):
        """
            Reset the internal state of the LSTM
        """
        self.h = None
        self.c = None

    def __call__(self, inputs, W, b):
        """
            Perform the LSTM op
            Args:
                inputs (float[][]): input tensor containing "x" to transform
        """
        x = inputs
        if self.h is not None:
            x += self.h_x(self.h, W, b)

        if self.c is None:
            self.c = variable.Variable(self.xp.zeros((len(inputs), self.n_units), dtype=self.xp.float32))

        self.c, self.h = F.lstm(self.c, x)
        return self.h
