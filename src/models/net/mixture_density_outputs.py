import numpy

from net.functions.mixture_density_outputs import mixture_density_outputs
from chainer.functions.connection import linear
from chainer import link
#from chainer.links.connection import linear
from chainer import variable
import chainer.functions

class MixtureDensityOutputs(link.Link):
    """Mixture-Density-Outputs layer.

    This is a Mixture-Density-Outputs layer as a chain. 
    This is a link that wraps the :func:`~chainer.functions.mixturedensityoutputs` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as parameters.
    

    Args:
        mix_size (int): number of mixture components.
        
    Attributes:
        
    """
    
    def __init__(self, mix_size):
        super(MixtureDensityOutputs, self).__init__()
        self.loss=None
        self.mix_size=mix_size
    
    def __call__(self, xnext, eow, h1, h2, h3, W1, W2, W3, b1, b2, b3, prob_bias):
        """
            xnext   :   next state of a pen. ndim=(batchsize,3)
            h       :   input vector 
            W1, W2, W3: (h.shape[1], 1 + mix_size * 6)
            b1, b2, b3: (1, 1 + mix_size * 6)
            prob_bias:  probability bias
        """
        
        mix_size = self.mix_size     
        y  = linear.linear(h1, W1, b1)
        y += linear.linear(h2, W2, b2)
        y += linear.linear(h3, W3, b3) 
        
        eos_hat, pi_hat, mu1_hat, mu2_hat, sg1_hat, sg2_hat, rho_hat = chainer.functions.split_axis(
                y,
                numpy.asarray([1, 1+mix_size, 1+2*mix_size, 1+3*mix_size, 1+4*mix_size, 1+5*mix_size]), axis=1)
        
        self.loss, self.xpred, self.eos, self.pi_, self.mux, self.muy, self.sgx, self.sgy, self.rho = mixture_density_outputs(
                            xnext, eow, eos_hat, pi_hat * (1. + prob_bias), mu1_hat, mu2_hat, sg1_hat - prob_bias, sg2_hat - prob_bias, rho_hat) 
        
        return self.loss, self.xpred, self.eos, self.pi_, self.mux, self.muy, self.sgx, self.sgy, self.rho
        
        
