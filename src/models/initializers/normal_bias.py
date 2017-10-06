import numpy as np

import chainer
from chainer import initializer
from chainer import cuda

class NormalBias(initializer.Initializer):
    """
        Initialize array with a normal distribution and bias.
        Mean is zero.
        Standard deviation is "scale".
        
        Args:
            out_size(int): Output size
            scale(float): Standard deviation of Gaussian distribution.
            bias(float): Inital bias value
            dtype: Data type specifier.
    """
    def __init__(self, out_size, scale=1., bias=0., dtype=None):
        self.out_size = out_size
        self.scale = scale
        self.bias = bias
        super(NormalBias, self).__init__(dtype)

    def __call__(self, array):
        # @NOTE We assume that array is 1D
        xp = cuda.get_array_module(array)
        in_size = array.shape[0]/self.out_size
        args = {'loc': 0.0, 'scale': self.scale, 'size': self.out_size*(in_size-1)}
        if xp is not np:
            # Only CuPy supports dtype option
            if self.dtype == np.float32 or self.dtype == np.float16:
                # float16 is not supported in cuRAND
                args['dtype'] = np.float32

        normal = xp.random.normal(**args).astype(self.dtype).reshape((1, args['size']))
        bias = xp.ones((1, self.out_size)).astype(self.dtype) * self.bias
        array[...] = xp.concatenate((normal, bias), axis=1).reshape(array.shape)
