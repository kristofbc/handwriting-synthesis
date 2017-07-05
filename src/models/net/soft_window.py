import numpy

from net.functions.soft_window import soft_window
from chainer.functions.connection import linear
from chainer import link
#from chainer.links.connection import linear
from chainer import variable
import chainer.functions

class SoftWindow(link.Link):
    """SoftWindow layer.

    This is a SoftWindow layer as a chain. 
    This is a link that wraps the :func:`~chainer.functions.mixturedensityoutputs` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as parameters.
    

    Args:
        mix_size (int): number of mixture components.
        
    Attributes:
        
    """
    
    def __init__(self, mix_size):
        super(SoftWindow, self).__init__()
        self.mix_size=mix_size
        self.reset_state()
        
    def reset_state(self):
        
        self.k_prev = None
            

    def __call__(self, cs, ls, h, W, b):
        """
            cs   :   one-hot-encoding of a length U character sequence 
            h    :   input vector (summation of affine transformation of outputs from hidden layers)
        """
        mix_size = self.mix_size
        y = linear.linear(h, W, b)
        
        a_hat, b_hat, k_hat =  chainer.functions.split_axis(y, numpy.asarray([mix_size, 2*mix_size]), axis=1) 
        
        if self.k_prev is None:
            xp = self.xp
            self.k_prev = variable.Variable(
                xp.zeros_like(k_hat.data),
                volatile='auto')
            
        self.ws, self.k_prev, self.eow = soft_window(cs, ls, a_hat, b_hat, k_hat, self.k_prev)
        return self.ws, self.eow
