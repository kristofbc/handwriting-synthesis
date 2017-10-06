import chainer
import chainer.functions as F
from chainer import variable

from links.connection.linear import Linear
from links.normalization.layer_normalization import LayerNormalization

class LinearLayerNormalizationLSTM(chainer.Chain):
    """
        Alex Graves' LSTM implementation with Linear activation and LayerNormalization
        Args:
            n_units (int): Number of units inside this LSTM
            forget_bias_init (float): bias added to the forget gate before sigmoid activation
            norm_bias_init (float): optional bias parameter
            norm_gain_init (float): optional gain parameter
    """
    def __init__(self, n_units, forget_bias_init = 0., norm_bias_init = 0., norm_gain_init = 1.):
        super(LinearLayerNormalizationLSTM, self).__init__()

        self.n_units = n_units
        self.forget_bias = forget_bias_init
        self.h = None
        self.c = None

        with self.init_scope():
            self.h_x = Linear(None, n_units, no_bias=True)
            self.norm_c = LayerNormalization(None, norm_bias_init, norm_gain_init)
            self.norm_x = LayerNormalization(None, norm_bias_init, norm_gain_init)
            self.norm_h = LayerNormalization(None, norm_bias_init, norm_gain_init)

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
        x = self.norm_x(x)
        if self.h is not None:
            x += F.bias(self.norm_h(self.h_x(self.h, W)), b)

        if self.c is None:
            self.c = variable.Variable(self.xp.zeros((len(inputs), self.n_units), dtype=self.xp.float32))

        # Compute the LSTM using Chainer's function to be able to use LayerNormalization
        def extract_gates(x):
            r = F.reshape(x, (x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
            return F.split_axis(r, 4, axis=2)

        a, i, f, o = extract_gates(x)
        # Remove unused dimension and apply transformation
        a = F.tanh(F.squeeze(a, axis=2))
        i = F.sigmoid(F.squeeze(i, axis=2))
        f = F.sigmoid(F.squeeze(f, axis=2) + self.forget_bias)
        o = F.sigmoid(F.squeeze(o, axis=2))

        # Transform
        c = a * i + f * self.c
        # Apply LayerNormalization
        h = o * F.tanh(self.norm_c(c))

        self.c, self.h = c, h
        return self.h
