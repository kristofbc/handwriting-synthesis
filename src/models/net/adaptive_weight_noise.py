import numpy

from net.functions.adaptive_weight_noise import adaptive_weight_noise
from chainer.functions.math import sum
from chainer.functions.array import reshape
from chainer.functions.array import broadcast
from chainer.functions.array import split_axis
from chainer import link


class AdaptiveWeightNoise(link.Link):

    def __init__(self, in_size, out_size, wscale=1, nobias=False, use_weight_noise=True):
        super(AdaptiveWeightNoise, self).__init__(
            M=(out_size*in_size)
        )
        self.add_param('logS2', (out_size*in_size))
        if nobias:
            self.M.data[...]  =  numpy.random.normal(0, wscale, (out_size*in_size)).astype(numpy.float32)
        else:
            self.M.data[...]  =  numpy.c_[numpy.random.normal(0, wscale, (out_size*(in_size-1))).astype(numpy.float32).reshape((1, out_size*(in_size-1))), numpy.zeros((1, out_size)).astype(numpy.float32)].reshape(out_size*in_size)
        self.logS2.data[...] =  numpy.log((numpy.ones((out_size*in_size))*1e-8).astype(numpy.float32))
        self.in_size  = in_size
        self.out_size = out_size
        self.nobias = nobias
        self.use_weight_noise = use_weight_noise
    
    
    def __call__(self, batchsize):
        """....
        
        Args:
            eps (~chainer.Variable): 
                a wsize-length vector whose elements are drawn from 
                normal distribution (mean = 0, std = 1).
            batchsize  (~chainer.Variable): 
                (batch size)  *  (number of truncated backward gradient calculation for a training dataset)

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        """
        self.m_hat  = reshape.reshape(sum.sum(self.M)/self.M.data.shape[0], (1,1))
        M, m_hat = broadcast.broadcast(self.M, self.m_hat)
        self.s2_hat = sum.sum(self.S2 + (M - m_hat)*(M - m_hat))/self.M.data.shape[0]
        
        print('m_hat.data {}'.format(self.m_hat.data))
        print('self.s2_hat.data {}'.format(self.s2_hat.data))
        print('self.S2.data {}'.format(self.S2.data))
        print('self.M.data {}'.format(self.M.data))
        print('------------------')
        """
        
        self.fWb, loss = adaptive_weight_noise(batchsize, self.M, self.logS2, self.use_weight_noise) 
        
        if self.nobias:
            return reshape.reshape(self.fWb, (self.out_size, self.in_size)), loss
        else:
            self.fW, self.fb = split_axis.split_axis(self.fWb, numpy.asarray([(self.in_size -1)*self.out_size]), axis=0)
            return reshape.reshape(self.fW, (self.out_size, self.in_size -1)), self.fb, loss
