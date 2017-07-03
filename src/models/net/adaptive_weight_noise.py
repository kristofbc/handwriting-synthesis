import numpy

from chainer.functions.math import sum
from chainer.functions.array import reshape
from chainer.functions.array import broadcast
from chainer.functions.array import split_axis
from chainer import link


class AdaptiveWeightNoise(link.Link):

    def __init__(self, in_size, out_size, batchsize, wscale=1, nobias=False, use_weight_noise=True):
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
	self.batchsize = batchsize
        self.nobias = nobias
	self.use_weight_noise = use_weight_noise
    
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        bs_type, M_type, logS2_type = in_types
        
    def forward(self, inputs):
	if self.use_weight_noise:
	    xp = cuda.get_array_module(*inputs)
	    bs, M, logS2 = inputs
	    self.m_hat  = xp.mean(M)
	    self.s2_hat = xp.mean(xp.exp(logS2) + (M - self.m_hat)**2)
	    #print(self.m_hat, self.s2_hat, xp.log(self.s2_hat), xp.min(logS2), xp.max(logS2))
	    #eps  = xp.random.randn(M.size,  dtype=xp.float32)
	    eps  = xp.random.randn(M.size).astype(xp.float32)
	    W    = M  + eps * xp.sqrt(xp.exp(logS2)) #* 0.001
	    loss = 0.5*(xp.log(self.s2_hat) - logS2) + 0.5*((M - self.m_hat)**2 + xp.exp(logS2) - self.s2_hat)/(self.s2_hat + 1e-8)
	    return W, xp.sum(loss)/bs
	else:
	    xp = cuda.get_array_module(*inputs)
	    bs, M, logS2 = inputs
	    #self.m_hat  = xp.mean(M)
	    #self.s2_hat = xp.mean(xp.exp(logS2) + (M - self.m_hat)**2)
	    #eps  = xp.random.randn(M.size,  dtype=xp.float32)
	    W    = M #+ eps * xp.sqrt(xp.exp(logS2))
	    loss = xp.zeros_like(M) #0.5*(xp.log(self.s2_hat) - logS2) + 0.5*((M - self.m_hat)**2 + xp.exp(logS2) - self.s2_hat)/(self.s2_hat + 1e-8)
	    return W, xp.sum(loss)/bs
        
    def backward(self, inputs, grad_outputs):
	if self.use_weight_noise:
	    xp = cuda.get_array_module(*inputs)
	    bs, M, logS2 = inputs
	    gW  = grad_outputs[0]
	    gM  = xp.zeros_like(M)
	    glogS2 = xp.zeros_like(logS2)
	    gM  =  (M - self.m_hat)/((self.s2_hat + 1e-8)*bs)        + gW
	    #gS2 = 0.5*(1/(self.s2_hat + 1e-8) - 1/(S2 + 1e-8))/bs + 0.5 * gW * gW 
	    glogS2 = 0.5*(xp.exp(logS2)/ self.s2_hat - 1.)/bs + 0.5 * xp.exp(logS2) * gW * gW 
	    #print(xp.min(gW), xp.max(gW), xp.min(gM), xp.max(gM), xp.min(glogS2), xp.max(glogS2)) 
	    return None, gM.astype(xp.float32), glogS2.astype(xp.float32)
	else:
	    xp = cuda.get_array_module(*inputs)
	    bs, M, logS2 = inputs
	    gW  = grad_outputs[0]
	    gM  = xp.zeros_like(M)
	    #glogS2 = xp.zeros_like(logS2)
	    gM  = gW # (M - self.m_hat)/((self.s2_hat + 1e-8)*bs)        + gW
	    #glogS2 = 0.5*(xp.exp(logS2)/ self.s2_hat - 1.)/bs + 0.5 * xp.exp(logS2) * gW * gW 
	    return None, gM, None
    
    def __call__(self):
        """....
        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        
        self.fWb, loss = super(AdaptiveWeightNoise, self).__call__(self.M, self.logS2) 
        
        if self.nobias:
            return reshape.reshape(self.fWb, (self.out_size, self.in_size)), loss
        else:
            self.fW, self.fb = split_axis.split_axis(self.fWb, numpy.asarray([(self.in_size -1)*self.out_size]), axis=0)
            return reshape.reshape(self.fW, (self.out_size, self.in_size -1)), self.fb, loss

