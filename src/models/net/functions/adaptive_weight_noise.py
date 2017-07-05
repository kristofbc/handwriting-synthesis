import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class AdaptiveWeightNoise(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        bs_type, M_type, logS2_type = in_types
        
    def forward(self, inputs):
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
        
    def backward(self, inputs, grad_outputs):
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

class NoAdaptiveWeightNoise(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        bs_type, M_type, logS2_type = in_types

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        bs, M, logS2 = inputs
        #self.m_hat  = xp.mean(M)
        #self.s2_hat = xp.mean(xp.exp(logS2) + (M - self.m_hat)**2)
        #eps  = xp.random.randn(M.size,  dtype=xp.float32)
        W    = M #+ eps * xp.sqrt(xp.exp(logS2))
        loss = xp.zeros_like(M) #0.5*(xp.log(self.s2_hat) - logS2) + 0.5*((M - self.m_hat)**2 + xp.exp(logS2) - self.s2_hat)/(self.s2_hat + 1e-8)
        return W, xp.sum(loss)/bs

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        bs, M, logS2 = inputs
        gW  = grad_outputs[0]
        gM  = xp.zeros_like(M)
        #glogS2 = xp.zeros_like(logS2)
        gM  = gW # (M - self.m_hat)/((self.s2_hat + 1e-8)*bs)        + gW
        #glogS2 = 0.5*(xp.exp(logS2)/ self.s2_hat - 1.)/bs + 0.5 * xp.exp(logS2) * gW * gW 
        return None, gM, None




def adaptive_weight_noise(batchsize, M, logS2, adaptive_noise=True):
    """Adaptive Weight Noise function.

    Args:
        batchsize (~chainer.Variable): 'total number of minibatches' variable of shape ``(1)``.
        
        M    (~chainer.Variable): 'mean of weight distribution' variable of shape ``(weight_size,)``.
        S2   (~chainer.Variable): 'variance of weight distribution' variable of shape ``(weight_size,)``..

    Returns:
        W    (~chainer.Variable): Precursor of Weight and Bias variable.
        loss (~chainer.Variable): Minimum Description length loss function for a batch
    

    """
    
    #return Adaptive_Weight_Noise()(batchsize, M, logS2)
    if adaptive_noise:
        return AdaptiveWeightNoise()(batchsize, M, logS2)
    else:
        return NoAdaptiveWeightNoise()(batchsize, M, logS2)
