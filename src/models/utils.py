import numpy as np

def clip_grad(value, th_min, th_max, batch_size=None, xp=np):
    """
        Clip a value between [th_min, th_max]
        @NOTE: Only th_max is supported for now

        Args:
            value (array[][]): the array containing the values to clip
            th_min (float): treshold min, lower bound
            th_max (float): treshold max, upper bound
            batch_size (int): output shape
            xp (numpy|cupy): array package
        Returns:
            value (array[][])
    """
    if batch_size is None:
        batch_size = value.shape[0]

    # @NOTE: Prevent nan?
    #value = xp.where(xp.isnan(value), 1e-10, value)
    #value = xp.clip(value, th_min, th_max)
    vmax = xp.max(xp.absolute(value), axis=1).reshape((batch_size, 1))
    rate = xp.where(vmax > th_max, th_max/vmax, 1.0).astype(xp.float32).reshape((batch_size, 1))
    return value * rate