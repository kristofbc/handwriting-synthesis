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

def get_max_sequence_length(sequences):
    length = []
    for i in xrange(len(sequences)):
        length.extend([sequences[i].size])
    return int(np.asarray(length).max())

def group_data(data, characters):
    if len(data) != len(characters):
        raise ValueError("data should have the same amount of characters")

    xp = cuda.get_array_module(*data)
    grouped = []
    for i in xrange(len(data)):
        grouped.append(xp.asarray([data[i], characters[i]]))

    return xp.vstack(grouped)

def pad_data(data, characters):
    max_length_data = 0
    max_length_characters = 0
    # Get the maximum length of each arrays
    tmp1 = []
    tmp2 = []
    for i in xrange(len(data)):
        if len(data[i]) > max_length_data:
            max_length_data = len(data[i])

    # Pad each arrays to be the same length
    for i in xrange(len(data)):
        if len(data[i]) != max_length_data:
            pad_length = max_length_data-len(data[i])
            pad = np.full((pad_length, 3), 0)
            pad[:, 2] = 2.
            data[i] = np.vstack([data[i], pad])
        tmp1.append(np.asarray(data[i]))
        tmp2.append(np.asarray(characters[i]))
    
    return np.asarray(tmp1), np.asarray(tmp2) 

def one_hot(data, characters, n_chars, n_max_seq_length):
    xp = cuda.get_array_module(*data)
    cs = xp.zeros((len(data), n_chars, n_max_seq_length), dtype=xp.float32)

    for i in xrange(len(data)):
        for j in xrange(len(characters[i])):
            k = characters[i][j]
            cs[i, k, j] = 1.0

    return cs

