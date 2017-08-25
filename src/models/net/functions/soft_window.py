import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _mat_ptrs(a):
    """Creates an array of pointers to matrices

    Args:
        a: A batch of matrices on GPU
    Returns:
        GPU array of pointers to matrices
    """
    if a.shape[0] == 1:
        return cuda.cupy.full((1,), a[0].data.ptr.value, dtype=numpy.intp)
    else:
        stride = a[1].data.ptr - a[0].data.ptr
        return cuda.cupy.arange(
            a[0].data.ptr,
            a[0].data.ptr + stride * a.shape[0],
            stride,
            dtype=numpy.intp)

def _as_batch_mat(x):
    return x.reshape((x.shape[0], x.shape[1], 1)) if len(x.shape) == 2 else x

def _get_ld(a):
    shape = a.shape[-2:]
    strides = a.strides[-2:]
    trans = numpy.argmin(strides)
    return trans, int(max(shape[trans], max(strides) // a.itemsize))

def _batch_matmul_gpu(a, b, out, transa=False, transb=False, transout=False):
    a = _as_batch_mat(a)
    b = _as_batch_mat(b)
    trans_axis = (0, 2, 1)
    if transout:
        out = out.transpose(trans_axis)
    needtrans, _ = _get_ld(out)
    if needtrans == 1:
        # (A B)^T = B^T A^T
        a, b = b, a
        transa, transb = not transb, not transa
        out = out.transpose(trans_axis)
    if transa:
        a = a.transpose(trans_axis)
    if transb:
        b = b.transpose(trans_axis)

    transa, lda = _get_ld(a)
    transb, ldb = _get_ld(b)
    transout, ldout = _get_ld(out)
    la, n, ka = a.shape
    lb, kb, m = b.shape

    assert ka == kb
    assert transout == 0 or ldout == 1
    assert out.shape == (la, n, m)

    ap = _mat_ptrs(a)
    bp = _mat_ptrs(b)
    outp = _mat_ptrs(out)
    cuda.cublas.sgemmBatched(
        cuda.Device().cublas_handle,
        transa,
        transb,
        n, m, ka, 1.0,
        ap.data.ptr, lda,
        bp.data.ptr, ldb,
        0.0, outp.data.ptr, ldout, la)


class SoftWindow(function.Function):

    """SoftWindow unit for handwriting synthesis (Graves 2013).

    It has five inputs (e_hat, pi_hat, mux_hat, muy_hat, sgmx_hat, sgmy_hat, rho_hat), 
    and five outputs (e, pi, mux, muy, sgmx, sgmy, rho), where e_hat and e are scalar,
    pi_hat, pi, mux_hat, muy_hat, sgmx_hat, sgmy_hat, mux, muy, sgmx, sgmy, 
    rho_hat and rho are M-length, 1 dimensional vectors.
    

    """
    """
    def __init__(self, K, initialkappa=None):
        
        if initialkappa is not None:
            assert initialkappa.shape ==(K, 1)
            self.kappa_prev = initialkappa
        else:
            self.kappa_prev = numpy.zeros((K, 1)).astype(numpy.float32)
    """    
        
    
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 6)
        cs_type, ls_type, alpha_type, beta_type, kappa_type, kappa_prev_type = in_types
        
                            
    def forward(self, inputs):
        cs, ls, alpha_hat, beta_hat, kappa_hat, kappa_prev = inputs
        
        # cs :  one-hot-encoding vectors whose shape is (W, U)
        #       U: maximal length of character sequences in a batch
        #       W: number of characters used in a data
        #
        # ls : a vector containing lengths of character sequences in a batch
        #
        # alpha, beta, kappa: length K vectors. shape = (batchsize, K)
        #  
             
        batchsize, W, U = cs.shape
        K = alpha_hat.shape[1]
         
        if isinstance(cs, numpy.ndarray):
            self.alpha = numpy.exp(alpha_hat).reshape((batchsize, K, 1))
            self.beta  = numpy.exp(beta_hat).reshape((batchsize, K, 1))
            self.kappa = (kappa_prev + numpy.exp(kappa_hat)).reshape((batchsize, K, 1))
            us = numpy.arange(U).astype(numpy.float32).reshape((1, 1, U))
            self.phai_mat =  self.alpha * numpy.exp(- self.beta * (self.kappa - us)**2) # --> (batchsize, K, U)
            ws = numpy.matmul(cs, self.phai_mat.sum(axis=1).reshape(batchsize, U, 1)) # (batchsize, W, U) x (batchsize, U, 1)--> (batchsize, W, 1)
            if ls.sum() > 0: #ls is not None:
                max_phai_idx = numpy.sum(self.phai_mat, axis=1).argmax(axis=1) # (batchsize, K, U) --> (batchsize, U) --> (batchsize, 1)
                eow = numpy.where(max_phai_idx > ls, max_phai_idx, -1)                # (batchsize, 1)
            else:
                eow = numpy.zeros((batchsize, U)) #None
        else:    
            self.alpha, self.beta, self.kappa = cuda.elementwise(
            'T a_hat, T b_hat, T ka_hat, T ka_prev', 
            'T a, T b, T ka',
            '''
                a = exp(a_hat);
                b = exp(b_hat);
                ka = ka_prev + exp(ka_hat);
            ''',
            'softwindow_fwd1'
            )(alpha_hat, beta_hat, kappa_hat, kappa_prev)
            
            us = cuda.cupy.arange(U).astype(cuda.cupy.float32).reshape((1, 1, U)) 
            
            self.phai_mat = cuda.elementwise(
            'T a, T b, T k, T u', 'T ph',
            '''
                ph = a * exp(- b *(k - u)*(k - u));
            ''',
            'softwindow_fwd2'
            )(
              self.alpha.reshape(batchsize, K, 1), 
              self.beta.reshape(batchsize, K, 1), 
              self.kappa.reshape(batchsize, K, 1), 
              us #cuda.cupy.arange(U).astype(cuda.cupy.float32).reshape((1, 1, U))
            )
            
            #phais = self.phai_mat.sum(axis=1).reshape(batchsize, U, 1)
            phais = cuda.reduce(
                'T x',
                'T y',
                'x',
                'a+b',
                'y=a',
                '0',
                'softwindow_fwd3',
            )(self.phai_mat, axis=1)
            if ls.sum() > 0: # ls is not None:
                max_phai_idx = cuda.cupy.argmax(phais, axis=1, keepdims=True)
                
            phais = phais.reshape(batchsize, U, 1)
            ws = cuda.cupy.empty((batchsize, W, 1)).astype(cuda.cupy.float32)
            _batch_matmul_gpu(cs, phais, out=ws)
            if ls.sum() > 0: # ls is not None:
                eow = cuda.cupy.where(max_phai_idx > ls, max_phai_idx, -1)
            else:
                eow = cuda.cupy.zeros((batchsize, U)) #None
                
        return ws.reshape(batchsize, W), self.kappa.reshape((batchsize, K)), eow

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        cs, ls, alpha_hat, beta_hat, kappa_hat, kappa_prev  = inputs
        
        batchsize, W, U = cs.shape
        K = alpha_hat.shape[1] 
        gw, gk = grad_outputs[0:2] # (batchsize, W)
        
        ga_hat = xp.empty_like(alpha_hat)
        gb_hat = xp.empty_like(beta_hat)
        gk_hat = xp.empty_like(kappa_hat)
        gk_prev = xp.empty_like(kappa_prev) #gk)
        gc = xp.empty_like(cs)

        # Consider the case that either gradient is not given
        if gw is None:
            gw = 0
        if gk is None:
            gk = 0
         
        if xp is numpy:
            gwc       = numpy.matmul(gw.reshape(batchsize, 1, W), cs) # (batchsize, 1, U)
            emat      = self.phai_mat * gwc # (batchsize, K, U)
            ga_hat[:] = emat.sum(axis=2)
            us        = numpy.arange(U).astype(numpy.float32).reshape((1, 1, U))
            diff      = us - self.kappa
            b         = self.beta.reshape((batchsize, K)) 
            gb_hat[:] = - b *(emat * diff**2).sum(axis=2)
            gk_prev[:]= gk + 2. * b * (emat * diff).sum(axis=2)
            gk_hat[:] = numpy.exp(kappa_hat)*gk_prev
        else:
            gwc = cuda.cupy.empty((batchsize, 1, U)).astype(cuda.cupy.float32)
            #for i in xrange(batchsize):
            #    gwc[i] = (gw.reshape(batchsize, 1, W))[i].dot(cs[i]) # (1, W).(W, U) --> (1, U)
            _batch_matmul_gpu(gw.reshape(batchsize, 1, W), cs, out=gwc) 
            #emat      = self.phai_mat * gwc
            
            emat = cuda.elementwise(
            'T phai, T gwc', 'T emat',
            '''
                emat = phai * gwc;
            ''',
            'softwindow_bw1',
            )(self.phai_mat, gwc)
            
            #ga_hat[:] = emat.sum(axis=2)
            
            ga_hat[:] = cuda.reduce(
                'T x',
                'T y',
                'x',
                'a+b',
                'y=a',
                '0',
                'softwindow_bw2',
            )(emat, axis=2)
            
            us        = cuda.cupy.arange(U).astype(cuda.cupy.float32).reshape((1, 1, U))
            diff      = us - self.kappa.reshape(batchsize, K, 1)
            b         = self.beta.reshape(batchsize, K)
            tmp2, tmp1 = cuda.elementwise(
            'T emat, T diff', 'T ed2, T ed1',
            '''
                ed1 = emat * diff;
                ed2 = ed1  * diff;
            ''',
            'softwindow_bw3'
            )(emat, diff)
            sum1 = cuda.reduce(
                'T x',
                'T y',
                'x',
                'a+b',
                'y=a',
                '0',
                'softwindow_bw4',
            )(tmp1, axis=2)
            sum2 = cuda.reduce(
                'T x',
                'T y',
                'x',
                'a+b',
                'y=a',
                '0',
                'softwindow_bw5',
            )(tmp2, axis=2)
            gb_hat[:] = - b * sum2
            gk_prev[:]= gk + 2. * b * sum1
            #gb_hat[:] = - b * (emat * diff**2).sum(axis=2)
            #gk_prev[:]= gk + 2. * b * (emat * diff).sum(axis=2)
            #gk_hat[:] = cuda.cupy.exp(kappa_hat)*gk_prev 
            gk_hat = cuda.elementwise(
            'T k_hat, T gk_prev',
            'T gk_hat',
            '''
                gk_hat = exp(k_hat)*gk_prev;
            ''',
            'softwindow_bw6'
            )(kappa_hat, gk_prev)
            
        return None, None, ga_hat, gb_hat, gk_hat, gk_prev, 


def soft_window(cs, ls, alpha_hat, beta_hat, kappa_hat, kappa_prev):
    
    return SoftWindow()(cs, ls, alpha_hat, beta_hat, kappa_hat, kappa_prev)
