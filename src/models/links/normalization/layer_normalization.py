import chainer
import chainer.functions as F
from chainer import variable

class LayerNormalization(chainer.Link):
    """
        Implementation of LayerNormalization from Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization.
        Args:
            hidden_size (int|shape): shape of the hidden size
            bias_init (float): optional bias parameter
            gain_init (float): optional gain parameter
            epsilon (float): computation stability parameters
    """
    def __init__(self, hidden_size, bias_init = 0., gain_init = 1., epsilon = 1e-6):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon

        with self.init_scope():
            self.bias = variable.Parameter(bias_init)
            self.gain = variable.Parameter(gain_init)

            if hidden_size is not None:
                self._initialize_params(hidden_size)

    def _initialize_params(self, size):
        self.bias.initialize(size)
        self.gain.initialize(size)
        self.hidden_size = size

    def __call__(self, x):
        """
            Apply the LayerNormalization on in the input "x"
            Args:
                x (float[][]): input tensor to re-center and re-scale (layer normalize)
            Returns:
                float[][]
        """
        if self.hidden_size is None:
            self._initialize_params(x.shape)

        # Layer Normalization parameters
        mu = F.average(x, axis=1, keepdims=True)
        mu = F.broadcast_to(mu, x.shape)
        sigma = F.sqrt(F.average(F.square(x - mu), axis=1, keepdims=True) + self.epsilon)
        sigma = F.broadcast_to(sigma, x.shape)

        # Transformation
        outputs = (x - mu) / sigma
        # Affine transformation
        outputs = (outputs * self.gain) + self.bias
        #outputs = F.scale(outputs, self.gain)
        #outputs = F.bias(outputs, self.bias)
        
        return outputs
