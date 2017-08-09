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
        
        
class MixtureDensityOutputs(function.Function):

    """Mixture-Densiy-Outputs unit for handwriting prediction/synthesis (Graves 2013).
    
    This function outputs Pr(x[t+1]|y[t]) where x[t] is a 3-dimensional vector (eg., x[t] = (x, y, z)). 

    It has five inputs (e_hat, pi_hat, mux_hat, muy_hat, sgmx_hat, sgmy_hat, rho_hat), 
    and five outputs (e, pi, mux, muy, sgmx, sgmy, rho), where e_hat and e are scalar,
    pi_hat, pi, mux_hat, muy_hat, sgmx_hat, sgmy_hat, mux, muy, sgmx, sgmy, 
    rho_hat and rho are M-length, 1 dimensional vectors.
    

    """
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 9)
        x_type, eow_type, e_type, pi_type, mux_type, muy_type, sgmx_type, sgmy_type, rho_type = in_types
        
        type_check.expect(
            x_type.dtype    == numpy.float32,
            e_type.dtype    == numpy.float32,

            x_type.ndim >= 2,
            e_type.ndim >= 2,

            x_type.shape[0] == e_type.shape[0],
        )
        
        for i in range(2, type_check.eval(x_type.ndim)):
            type_check.expect(e_type.shape[i] == x_type.shape[i])
        
        type_check.expect(
            x_type.dtype    == numpy.float32,
            pi_type.dtype   == numpy.float32,

            x_type.ndim >= 2,
            pi_type.ndim >= 2,

            x_type.shape[0] == pi_type.shape[0],
        )
        for i in range(2, type_check.eval(x_type.ndim)):
            type_check.expect(pi_type.shape[i] == x_type.shape[i])
            
        type_check.expect(
            x_type.dtype    == numpy.float32,
            mux_type.dtype  == numpy.float32,
            
            x_type.ndim >= 2,
            mux_type.ndim >= 2,
            
            x_type.shape[0] == mux_type.shape[0],
        )
        for i in range(2, type_check.eval(x_type.ndim)):
            type_check.expect(mux_type.shape[i] == x_type.shape[i])
        
        type_check.expect(
            x_type.dtype    == numpy.float32,
            muy_type.dtype  == numpy.float32,
            
            x_type.ndim >= 2,
            muy_type.ndim >= 2,
            
            x_type.shape[0] == muy_type.shape[0],
        )
        for i in range(2, type_check.eval(x_type.ndim)):
            type_check.expect(muy_type.shape[i] == x_type.shape[i])
        
        type_check.expect(
            x_type.dtype    == numpy.float32,
            sgmx_type.dtype  == numpy.float32,
            
            x_type.ndim >= 2,
            sgmx_type.ndim >= 2,
            
            x_type.shape[0] == sgmx_type.shape[0],
        )
        for i in range(2, type_check.eval(x_type.ndim)):
            type_check.expect(sgmx_type.shape[i] == x_type.shape[i])
        
        type_check.expect(
            x_type.dtype    == numpy.float32,
            sgmy_type.dtype  == numpy.float32,
            
            x_type.ndim >= 2,
            sgmy_type.ndim >= 2,
            
            x_type.shape[0] == sgmy_type.shape[0],
        )
        for i in range(2, type_check.eval(x_type.ndim)):
            type_check.expect(sgmy_type.shape[i] == x_type.shape[i])
            
        type_check.expect(
            x_type.dtype    == numpy.float32,
            rho_type.dtype  == numpy.float32,
            
            x_type.ndim >= 2,
            rho_type.ndim >= 2,
            
            x_type.shape[0] == rho_type.shape[0],
        )
        for i in range(2, type_check.eval(x_type.ndim)):
            type_check.expect(rho_type.shape[i] == x_type.shape[i])
            
        type_check.expect(
            pi_type.dtype   == numpy.float32,
            mux_type.dtype  == numpy.float32,

            pi_type.ndim >= 2,
            mux_type.ndim >= 2,

            pi_type.shape[1] == mux_type.shape[1],
        )
        for i in range(2, type_check.eval(pi_type.ndim)):
            type_check.expect(mux_type.shape[i] == pi_type.shape[i])
        
        type_check.expect(
            pi_type.dtype   == numpy.float32,
            muy_type.dtype  == numpy.float32,

            pi_type.ndim >= 2,
            muy_type.ndim >= 2,

            pi_type.shape[1] == muy_type.shape[1],
        )
        for i in range(2, type_check.eval(pi_type.ndim)):
            type_check.expect(muy_type.shape[i] == pi_type.shape[i])
        
        type_check.expect(
            pi_type.dtype   == numpy.float32,
            sgmx_type.dtype  == numpy.float32,

            pi_type.ndim >= 2,
            sgmx_type.ndim >= 2,

            pi_type.shape[1] == sgmx_type.shape[1],
        )
        for i in range(2, type_check.eval(pi_type.ndim)):
            type_check.expect(sgmx_type.shape[i] == pi_type.shape[i])
        
        type_check.expect(
            pi_type.dtype   == numpy.float32,
            sgmy_type.dtype  == numpy.float32,

            pi_type.ndim >= 2,
            sgmy_type.ndim >= 2,

            pi_type.shape[1] == sgmy_type.shape[1],
        )
        for i in range(2, type_check.eval(pi_type.ndim)):
            type_check.expect(sgmy_type.shape[i] == pi_type.shape[i])
        
        type_check.expect(
            pi_type.dtype   == numpy.float32,
            rho_type.dtype  == numpy.float32,

            pi_type.ndim >= 2,
            rho_type.ndim >= 2,

            pi_type.shape[1] == rho_type.shape[1],
        )
        for i in range(2, type_check.eval(pi_type.ndim)):
            type_check.expect(rho_type.shape[i] == pi_type.shape[i])
                    
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        xnext, eow, e_hat, pi_hat, mux_hat, muy_hat, sgmx_hat, sgmy_hat, rho_hat = inputs
        batchsize, M = pi_hat.shape
        x1 = xnext[:,0].reshape((batchsize, 1))
        x2 = xnext[:,1].reshape((batchsize, 1))
        x3 = xnext[:,2].reshape((batchsize, 1))
        if isinstance(mux_hat, numpy.ndarray):
            self.x       = xnext
            self.eos     = 1./(1. + numpy.exp(e_hat)) #_sigmoid(e_hat)
            self.pi_     = numpy.exp(pi_hat)/numpy.exp(pi_hat).sum(axis=1).reshape((batchsize,1))
            self.mux     = mux_hat
            self.muy     = muy_hat
            self.sgmx    = numpy.exp(sgmx_hat)
            self.sgmy    = numpy.exp(sgmy_hat)
            self.rho_    = numpy.tanh(rho_hat)
            
            if x3.sum() >= 0.0: #xnext is not None: # training & validation
                #x1 = xnext[:,0].reshape((batchsize, 1))
                #x2 = xnext[:,1].reshape((batchsize, 1))
                #x3 = xnext[:,2].reshape((batchsize, 1))
                
                dx1 = (x1 - self.mux)/self.sgmx
                dx2 = (x2 - self.muy)/self.sgmy
                self.Zs  = dx1*dx1 + dx2*dx2 - 2.*self.rho_*dx1*dx2
                Ns = numpy.exp(- 0.5*self.Zs/ (1.-self.rho_**2))/(2.* 3.1415927 * self.sgmx * self.sgmy * numpy.sqrt(1. - self.rho_**2)+1e-10)
                gamma_hats =  self.pi_*Ns
                sum_gamma_hats = gamma_hats.sum(axis=1).reshape((batchsize, 1)) + 1e-10
                self.gammas = gamma_hats/sum_gamma_hats
                loss_t =  -numpy.log(sum_gamma_hats) - x3*numpy.log(self.eos) - (1. - x3)*numpy.log(1. - self.eos)
                idx = numpy.where(x3==2)[0]
                self.update_or_not = numpy.ones_like(x3)
                self.update_or_not[idx,0] = 0.0
                loss_t = loss_t * self.update_or_not
                self.xnext = xnext
            else:   # prediction
                xnext = numpy.zeros((batchsize, 3))
                myux_min = mux_hat.min(axis=1).reshape((batchsize, 1))
                myux_max = mux_hat.max(axis=1).reshape((batchsize, 1))
                myuy_min = muy_hat.min(axis=1).reshape((batchsize, 1))
                myuy_max = muy_hat.max(axis=1).reshape((batchsize, 1))
                protect_mask = numpy.ones((batchsize, 1))
                while protect_mask.sum() >0:
                    z1 = numpy.random.uniform(size=batchsize).reshape((batchsize, 1))
                    z2 = numpy.random.uniform(size=batchsize).reshape((batchsize, 1))
                    x1 = myux_min + (myux_max - myux_min) * z1
                    x2 = myuy_min + (myuy_max - myuy_min) * z2
                    
                    dx1 = (x1 - self.mux)/self.sgmx
                    dx2 = (x2 - self.muy)/self.sgmy
                    self.Zs  = dx1*dx1 + dx2*dx2 - 2.*self.rho_*dx1*dx2
                    Ns = numpy.exp(- 0.5*self.Zs/ (1.-self.rho_**2))/(2.* 3.1415927 * self.sgmx * self.sgmy * numpy.sqrt(1. - self.rho_**2)+1e-10)
                    gamma_hats =  self.pi_*Ns
                    sum_gamma_hats = gamma_hats.sum(axis=1) # Pr(x|ys)
                    
                    us = numpy.random.uniform(size=batchsize)
                    idx = numpy.where(sum_gamma_hats > us)[0]
                    xnext[idx, 0] += (x1*protect_mask)[idx, 0] 
                    xnext[idx, 1] += (x2*protect_mask)[idx, 0]
                    protect_mask[idx, 0] = 0.0
                
                xnext[:, 2] = self.eos[:, 0]
                #xnext[:, 2] = numpy.where(eow < 0, xnext[:, 2], 2.)
                xnext[:, 2] = numpy.where(eow[:, 2] < 0, xnext[:, 2], 2.)
                self.xnext = xnext
                #loss_t = None
                loss_t = xp.zeros((batchsize, 1)).astype(xp.float32)
                self.Zs = None
                    
        else:
            self.mux   = mux_hat
            self.muy   = muy_hat
            self.pi_hat     = pi_hat - pi_hat.max(axis=1).reshape(batchsize, 1)
            sum_exp_pi = cuda.reduce(
                'T x',      # input params
                'T y',      # output params
                'exp(x)',   # map
                'a+b',      # reduce
                'y=a',      # post-reduction map
                '1e-10',        # identity value
                'mdout_sumexp'    # kernel name
            )(self.pi_hat, axis=1)
            
            self.eos = 1./(1. + cuda.cupy.exp(e_hat))
            
            if x3.sum() >= 0.0: #xnext is not None:  # training & validation
                gamma_hats, self.Zs, self.pi_, self.sgmx, self.sgmy, self.rho_ = cuda.elementwise(
                'T x1, T x2, T pi_hat, T mux_, T muy_, T sgmx_hat, T sgmy_hat, T rho_hat, T sum_exp_pi', # input
                'T gammas, T Zs, T pi_, T sgmx_, T sgmy_, T rho_',                                      # output
                '''
                    pi_ = exp(pi_hat)/sum_exp_pi;
                    sgmx_ = exp(sgmx_hat) + 1e-10;
                    sgmy_ = exp(sgmy_hat) + 1e-10;
                    rho_ = tanh(rho_hat);
                    T rho2 = 1. - rho_*rho_ + 1e-10;
                    T dx1 = (x1 - mux_)/sgmx_;
                    T dx2 = (x2 - muy_)/sgmy_;
                    Zs = dx1*dx1 + dx2*dx2- 2.*rho_*dx1*dx2;
                    T Ns = exp( -0.5*Zs /rho2)/(2. * 3.1415927 * sgmx_ * sgmy_ * sqrt(rho2));
                    gammas = pi_ * Ns;
                ''',
                'mdout_fwd1', 
                )(x1, x2, self.pi_hat, mux_hat, muy_hat, sgmx_hat, sgmy_hat, rho_hat, sum_exp_pi.reshape((batchsize, 1)))
            
                sum_gamma_hats = gamma_hats.sum(axis=1).reshape((batchsize, 1)) + 1e-10
                self.gammas = gamma_hats/sum_gamma_hats
                loss_t = cuda.elementwise(
                'T sum_, T x3, T eos',
                'T loss',
                '''
                    loss = -log(sum_) - x3 * log(eos) - (1. - x3) * log(1.-eos);
                ''',
                'mdout_fwd2',
                )(sum_gamma_hats, x3, self.eos)
                self.update_or_not = xp.where(x3==2., 0.0, 1.0).astype(xp.float32)
                loss_t = loss_t * self.update_or_not
                self.xnext = xnext
                
            else:   # prediction (sampling from probability distribution)
                # pi, sgmx, sgmy, rho  <-- pi_hat, sgmx_hat, sgmy_hat, rho_hat
                self.pi_, self.sgmx, self.sgmy, self.rho_ = cuda.elementwise(
                'T pi_hat, T sgmx_hat, T sgmy_hat, T rho_hat, T sum_exp_pi', # input
                'T pi_, T sgmx_, T sgmy_, T rho_',                           # output
                '''
                    pi_ = exp(pi_hat)/sum_exp_pi;
                    sgmx_ = exp(sgmx_hat) + 1e-10;
                    sgmy_ = exp(sgmy_hat) + 1e-10;
                    rho_ = tanh(rho_hat);
                ''',
                'mdout_fwd3', 
                )(self.pi_hat, sgmx_hat, sgmy_hat, rho_hat, sum_exp_pi.reshape((batchsize, 1)))
                
                # because variances of gaussians are very small, sampling is virtually impossible, we set lower boundary for variances! 
                self.sgmx = xp.where( self.sgmx < 0.0015, 0.0015, self.sgmx)
                self.sgmy = xp.where( self.sgmy < 0.0015, 0.0015, self.sgmy)
                #print(self.sgmx.min(), self.sgmy.min())
                
                # get the (aproximated) maximum p value of M-mixture gaussian distributions. 
                # Here I assume that the maximum p value is taken at a center of a gaussian component in the mixture.
                # First, calculate p-values at each center of gaussian components,
                # and the maximum of these p-values is considered as the upper boundary of the M-mixture gaussian distributions 
                
                # prepare x1 and x2 matrices like
                # [ [mux0, mux0, ...., mux0],
                #   [mux1, mux1, ...., mux1],
                #   ...
                #   [muxn, muxn, ...., muxn]]  where n = batchsize
                
                muxs = xp.empty((batchsize, M, M)).astype(xp.float32)
                muys = xp.cupy.empty((batchsize, M, M)).astype(xp.float32)
                _batch_matmul_gpu(mux_hat.reshape((batchsize, M, 1)), xp.ones((batchsize, 1, M)).astype(xp.float32), out=muxs)
                _batch_matmul_gpu(muy_hat.reshape((batchsize, M, 1)), xp.ones((batchsize, 1, M)).astype(xp.float32), out=muys)
                
                # N_i((mux[j], muy[j])) for i = 0, 1, ..., M and j = 0, 1, ..., M
                gamma_hats_at_components = cuda.elementwise(
                'T x1, T x2, T pi_, T mux_, T muy_, T sgmx_, T sgmy_, T rho_',  # input
                'T gammas',                                                     # output
                '''
                    T rho2 = 1. - rho_*rho_ + 1e-10;
                    T dx1 = (x1 - mux_)/sgmx_;
                    T dx2 = (x2 - muy_)/sgmy_;
                    T Zs = dx1*dx1 + dx2*dx2- 2.*rho_*dx1*dx2;
                    T Ns = exp( -0.5*Zs /rho2)/(2. * 3.1415927 * sgmx_ * sgmy_ * sqrt(rho2));
                    gammas = pi_ * Ns;
                ''',
                'mdout_fwd5', 
                )(muxs, 
                  muys, 
                  self.pi_.reshape((batchsize, 1, M)), 
                  mux_hat.reshape((batchsize, 1, M)), 
                  muy_hat.reshape((batchsize, 1, M)), 
                  self.sgmx.reshape((batchsize, 1, M)), 
                  self.sgmy.reshape((batchsize, 1, M)), 
                  self.rho_.reshape((batchsize, 1, M))
                )
                
                # p[j] = sum(N_i((mux[j], muy[j])) for i = 0, 1, ..., M
                sum_gamma_hats_at_components = gamma_hats_at_components.sum(axis=2)        # (batchsize, M)
                # max(p[0], p[1], ..., p[M]) for each batch
                p_maxs = sum_gamma_hats_at_components.max(axis=1).reshape((batchsize, 1))  # (batchsize, 1)
                #print(p_maxs.reshape((1, batchsize)))
                
                myux_min = mux_hat.min(axis=1).reshape((batchsize, 1, 1)) - 0.01
                myux_max = mux_hat.max(axis=1).reshape((batchsize, 1, 1)) + 0.01
                myuy_min = muy_hat.min(axis=1).reshape((batchsize, 1, 1)) - 0.01
                myuy_max = muy_hat.max(axis=1).reshape((batchsize, 1, 1)) + 0.01
                
                xnext = xp.zeros((batchsize, 3)).astype(xp.float32)
                protect_mask = xp.ones((batchsize, 1)).astype(xp.float32)
                n_samples = 32768 * 2 #16384 #8192 #4096 #2048 #1024 #512
                while protect_mask.sum() >0:
                    # sampling n (=n_samples) samples in parallel at a step
                    z1 = xp.random.uniform(size=batchsize* n_samples).reshape((batchsize, n_samples, 1))
                    z2 = xp.random.uniform(size=batchsize* n_samples).reshape((batchsize, n_samples, 1))
                    x1_ = (myux_min + (myux_max - myux_min) * z1).astype(xp.float32)  # (batchsize, n_samples, 1)
                    x2_ = (myuy_min + (myuy_max - myuy_min) * z2).astype(xp.float32)  # (batchsize, n_samples, 1)
                    gamma_hats = cuda.elementwise(
                    'T x1, T x2, T pi_, T mux_, T muy_, T sgmx_, T sgmy_, T rho_',  # input
                    'T gammas',                                               # output
                    '''
                        T rho2 = 1. - rho_*rho_ + 1e-10;
                        T dx1 = (x1 - mux_)/sgmx_;
                        T dx2 = (x2 - muy_)/sgmy_;
                        T Zs = dx1*dx1 + dx2*dx2- 2.*rho_*dx1*dx2;
                        T Ns = exp( -0.5*Zs /rho2)/(2. * 3.1415927 * sgmx_ * sgmy_ * sqrt(rho2));
                        gammas = pi_ * Ns;
                    ''',
                    'mdout_fwd4', 
                    )(
                    x1_, x2_, 
                    self.pi_.reshape(( batchsize, 1, M)), 
                    mux_hat.reshape((  batchsize, 1, M)), 
                    muy_hat.reshape((  batchsize, 1, M)), 
                    self.sgmx.reshape((batchsize, 1, M)), 
                    self.sgmy.reshape((batchsize, 1, M)), 
                    self.rho_.reshape((batchsize, 1, M))
                    )
                    sum_gamma_hats_ = gamma_hats.sum(axis=2)
                    """
                    sum_gamma_hats  = sum_gamma_hats_.max(axis=1).reshape((batchsize, 1))
                    sample_idx = sum_gamma_hats_.argmax(axis=1).reshape((batchsize, 1))
                    for bb in xrange(batchsize):
                        this_midx = sample_idx[bb, 0]
                        x1[bb:bb+1, 0] = x1_[bb:bb+1, this_midx:this_midx+1, 0]
                        x2[bb:bb+1, 0] = x2_[bb:bb+1, this_midx:this_midx+1, 0]
                    us = xp.random.uniform(size=batchsize).reshape((batchsize, 1)) * p_maxs
                    update_mask  = xp.where(sum_gamma_hats > us, 1.0, 0.0).astype(xp.float32).reshape((batchsize, 1))
                    xnext[:, 0]  += (x1*protect_mask*update_mask)[:, 0]
                    xnext[:, 1]  += (x2*protect_mask*update_mask)[:, 0]
                    protect_mask -= protect_mask * update_mask
                    """
                    """
                    us_ = xp.random.uniform(size=batchsize* n_samples).reshape((batchsize, n_samples)) * p_maxs
                    update_mask_  = xp.where(sum_gamma_hats_ > us_, 1.0, 0.0).astype(xp.float32).reshape((batchsize, n_samples))
                    x1 = x1_.reshape((batchsize, n_samples)) * update_mask_
                    x2 = x2_.reshape((batchsize, n_samples)) * update_mask_
                    for i in xrange(n_samples):
                        xnext[:, 0]  += (x1_[:,i, :]*protect_mask)[:, 0]
                        xnext[:, 1]  += (x2_[:,i, :]*protect_mask)[:, 0]
                        #print(protect_mask.shape, update_mask_[:, i:(i+1)].shape)
                        protect_mask -= protect_mask * update_mask_[:, i:(i+1)]
                    """
                    us_ = xp.random.uniform(size=batchsize* n_samples).reshape((batchsize, n_samples)) * p_maxs
                    update_mask_  = xp.where(sum_gamma_hats_ > us_, 1.0, 0.0).astype(xp.float32).reshape((batchsize, n_samples))
                    update_mask = update_mask_.max(axis=1).reshape((batchsize, 1))
                    sample_idx  = update_mask_.argmax(axis=1).reshape((batchsize, 1))
                    for bb in xrange(batchsize):
                        this_midx = sample_idx[bb, 0]
                        x1[bb:bb+1, 0] = x1_[bb:bb+1, this_midx:this_midx+1, 0]
                        x2[bb:bb+1, 0] = x2_[bb:bb+1, this_midx:this_midx+1, 0]
                    xnext[:, 0]  += (x1*protect_mask*update_mask)[:, 0]
                    xnext[:, 1]  += (x2*protect_mask*update_mask)[:, 0]
                    protect_mask -= protect_mask * update_mask
                    
                    
                xnext[:, 2:] = self.eos[:, 0:1]
                #xnext[:, 2:] = xp.where(eow < 0, self.eos[:, 0:1], 2.)
                xnext[:, 2:] = xp.where(eow[:, 2:] < 0, self.eos[:, 0:1], 2.)
                self.xnext = xnext
                loss_t = xp.zeros((batchsize, 1)).astype(xp.float32)
                self.Zs = None
                
        return loss_t, self.xnext, self.eos, self.pi_, self.mux, self.muy, self.sgmx, self.sgmy, self.rho_,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        xnext, eow, e_hat, pi_hat, mux_hat, muy_hat, sgmx_hat, sgmy_hat, rho_hat  = inputs
        batchsize, M = pi_hat.shape
        x1 = xnext[:,0].reshape((batchsize, 1))
        x2 = xnext[:,1].reshape((batchsize, 1))
        x3 = xnext[:,2].reshape((batchsize, 1))
        
        #gpi,  = grad_outputs
        
        gpi = xp.empty_like(pi_hat)
        gmux = xp.empty_like(mux_hat)
        gmuy = xp.empty_like(muy_hat)
        gsgmx = xp.empty_like(sgmx_hat)
        gsgmy = xp.empty_like(sgmy_hat)
        grho = xp.empty_like(rho_hat)
        geos = xp.empty_like(e_hat)
        
        gxs = xp.zeros_like(xnext)

        # Consider the case that either gradient is not given
        if gpi is None:
            gpi = 0
        if gmux is None:
            gmux = 0
        if gmuy is None:
            gmuy = 0
        if gsgmx is None:
            gsgmx = 0
        if gsgmy is None:
            gsgmy = 0
        if grho is None:
            grho = 0
        if geos is None:
            geos = 0

        if xp is numpy:
            #update_or_not = xp.ones_like(x3)
            #idx = numpy.where(x3==2)[0]
            #update_or_not[idx,0] = 0.0
            C_    = 1./(1. - self.rho_*self.rho_)
            gpi   = (self.pi_ - self.gammas) * self.update_or_not
            dx1 = (x1 - self.mux)/self.sgmx
            dx2 = (x2 - self.muy)/self.sgmy
            CA1_   = C_*( dx1 - self.rho_*dx2 ) /self.sgmx 
            CA2_   = C_*( dx2 - self.rho_*dx1 ) /self.sgmy
            gmux  = - self.gammas * CA1_ * self.update_or_not
            gmuy  = - self.gammas * CA2_ * self.update_or_not
            gsgmx = - self.gammas * ( CA1_ * (x1 - self.mux) - 1.) * self.update_or_not
            gsgmy = - self.gammas * ( CA2_ * (x2 - self.muy) - 1.) * self.update_or_not
            grho  = - self.gammas * ( dx1*dx2 + self.rho_ * (1. - C_ * self.Zs) ) * self.update_or_not
            geos  = (x3 - self.eos) * self.update_or_not
        else:
            #update_or_not = xp.where(x3==2., 0.0, 1.0).astype(xp.float32)
            gpi, gmux, gmuy, gsgmx, gsgmy, grho = cuda.elementwise(
            'T x1, T x2, T gammas, T pi_, T mux, T muy, T sgmx, T sgmy, T rho_, T Zs, T un',
            'T gpi, T gmux, T gmuy, T gsgmx, T gsgmy, T grho',
            '''
                T C_ = 1. / (1. - rho_ * rho_ + 1e-10);
                T dx1 = (x1 - mux)/sgmx;
                T dx2 = (x2 - muy)/sgmy;
                T CA1 = C_ * ( dx1 - rho_*dx2 ) /sgmx;
                T CA2 = C_ * ( dx2 - rho_*dx1 ) /sgmy;
                gpi   = (pi_ - gammas) * un;
                gmux  = - gammas * CA1 * un;
                gmuy  = - gammas * CA2 * un;
                gsgmx = - gammas * ( CA1 * (x1 - mux) - 1.) * un;
                gsgmy = - gammas * ( CA2 * (x2 - muy) - 1.) * un;
                grho  = - gammas * ( dx1*dx2 + rho_*(1. - C_ * Zs)) * un;
            ''',
            'mdout_bwd',
            )(x1, x2, self.gammas, self.pi_, self.mux, self.muy, self.sgmx, self.sgmy, self.rho_, self.Zs,  self.update_or_not)
            
            geos = (x3 - self.eos) * self.update_or_not #* 4.0 #* 1000.0
            
        th_min = -100.0
        th_max =  100.0 

        geos_max = xp.max(xp.absolute(geos), axis=1).reshape((batchsize, 1))
        rate = xp.where(geos_max > th_max, th_max/geos_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        geos *=  rate
        
        gpi_max = xp.max(xp.absolute(gpi), axis=1).reshape((batchsize, 1))
        rate = xp.where(gpi_max > th_max, th_max/gpi_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        gpi *=  rate
        
        gmux_max = xp.max(xp.absolute(gmux), axis=1).reshape((batchsize, 1))
        rate = xp.where(gmux_max > th_max, th_max/gmux_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        gmux *=  rate
        
        gmuy_max = xp.max(xp.absolute(gmuy), axis=1).reshape((batchsize, 1))
        rate = xp.where(gmuy_max > th_max, th_max/gmuy_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        gmuy *=  rate
        
        gsgmx_max = xp.max(xp.absolute(gsgmx), axis=1).reshape((batchsize, 1))
        rate = xp.where(gsgmx_max > th_max, th_max/gsgmx_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        gsgmx *=  rate
        
        gsgmy_max = xp.max(xp.absolute(gsgmy), axis=1).reshape((batchsize, 1))
        rate = xp.where(gsgmy_max > th_max, th_max/gsgmy_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        gsgmy *=  rate
        
        grho_max = xp.max(xp.absolute(grho), axis=1).reshape((batchsize, 1))
        rate = xp.where(grho_max > th_max, th_max/grho_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        grho *=  rate
            
        #return gxs, geos.clip(th_min, th_max), gpi.clip(th_min, th_max), gmux.clip(th_min, th_max), gmuy.clip(th_min, th_max), gsgmx.clip(th_min, th_max),  gsgmy.clip(th_min, th_max), grho.clip(th_min, th_max),
        #print('mdn', geos, gpi, gmux, gmuy, gsgmx,  gsgmy, grho)
    #return  None, None, geos.clip(th_min, th_max), gpi.clip(th_min, th_max), gmux.clip(th_min, th_max), gmuy.clip(th_min, th_max), gsgmx.clip(th_min, th_max),  gsgmy.clip(th_min, th_max), grho.clip(th_min, th_max),
        return None, None, geos, gpi, gmux, gmuy, gsgmx,  gsgmy, grho,

def mixture_density_outputs(xnext, eow, e_hat, pi_hat, mux_hat, muy_hat, sgmx_hat, sgmy_hat, rho_hat):

    return MixtureDensityOutputs()(xnext, eow, e_hat, pi_hat, mux_hat, muy_hat, sgmx_hat, sgmy_hat, rho_hat)
