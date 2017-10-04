import numpy as np
from chainer import cuda

def clip_grad(value, th_min, th_max, xp=np):
    """
        Clip a value between [th_min, th_max]
        @NOTE: Only th_max is supported for now

        Args:
            value (array[][]): the array containing the values to clip
            th_min (float): treshold min, lower bound
            th_max (float): treshold max, upper bound
            xp (numpy|cupy): array package
        Returns:
            value (array[][])
    """
    # @NOTE: Prevent nan?
    #value = xp.where(xp.isnan(value), 1e-10, value)
    #value = xp.clip(value, th_min, th_max)
    res = xp.zeros_like(value)
    return value
    if xp == np:
        #vmax = xp.max(xp.absolute(value), axis=1).reshape((batch_size, 1))
        #rate = xp.where(vmax > th_max, th_max/vmax, 1.0).astype(xp.float32).reshape((batch_size, 1))
        #res = value * rate
        vmax = xp.absolute(value) + 1e-10
        rate = xp.where(vmax > th_max, th_max/vmax, 1.0).astype(xp.float32).reshape(value.shape)
        res = value * rate
    else:
        # GPU clipping
        res = cuda.elementwise(
            'T value, T th_min, T th_max',
            'T res',
            '''
                T vmax = fabsf(value) + 1e-10;
                T rate = (vmax>th_max) ? th_max / vmax : 1.0;
                res = value * rate;
            ''',
            'clip_grad_kern')(value, th_min, th_max)

        #if xp.all(xp.isnan(res)):
        #    raise ValueError("NaN detected")
        
    return res

def mean_squared_error(true, pred):
    """
        Mean square error between the prediction and ground-truth
        Args:
            pred (float[][]): prediction
            true (float[][]): ground-truth
        Returns:
            float|float[]
    """
    xp = cuda.get_array_module(*pred)
    return xp.sum(xp.square(true - pred)) / pred.size
