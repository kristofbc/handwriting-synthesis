import chainer
import chainer.functions as F

class Linear(chainer.Link):
    """
        Linear link with custom weight and bias initialization
        
        Args:
            in_size (int): Dimension of input vector
            out_size (tuple): Dimension of output vector
        Returns:
            float[][]
    """
    def __init__(self, in_size, out_size, no_bias=False):
        super(Linear, self).__init__()

        self.out_size = out_size
        self.in_size = in_size
        self.no_bias = no_bias

        with self.init_scope():
            self.W = chainer.Parameter(chainer.initializers.Normal(0.075))
            self.b = chainer.Parameter(chainer.initializers.Normal(0.075))

            # Initialization is on call (don't initialize unecessary params if they're given each time
            #if in_size is not None:
                #self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.in_size = in_size
        self.W.initialize((self.out_size, in_size))
        self.b.initialize((self.out_size))

    def __call__(self, x, W=None, b=None):
        """
            Perform the Linear operation with custom weights and bias

            Args:
                x (float[][]): input tensor "x" to transform
                W (float[][]): input weights
                b (float[]): input bias
            Returns
                float[][]
        """
        if W is None and b is None and self.W.data is None:
            if self.in_size is None:
                self._initialize_params(x.size // x.shape[0])
            else:
                self._initialize_params(self.in_size)

        if W is None:
            W = self.W
        if b is None:
            b = self.b

        if not self.no_bias:
            return F.linear(x, W, b)
        else:
            return F.linear(x, W)
