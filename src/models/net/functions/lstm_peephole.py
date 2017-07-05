import numpy
import math

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
    return (r[:, :, i] for i in range(4))

def _extract_gates_ph(x):
    r = x.reshape((x.shape[0], x.shape[1] // 3, 3) + x.shape[2:])
    return (r[:, :, i] for i in range(3))

def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); 
'''


class LSTMPeephole(function.Function):

    """Long short-term memory unit with forget gate.

    It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
    state. x must have four times channels compared to the number of units.

    """
    def check_type_forward(self, in_types):
        #type_check.expect(in_types.size() == 2)
        n_in = in_types.size()
        type_check.expect(3 <= n_in, n_in <= 4)
        c_type, x_type, w_type = in_types[:3]

        type_check.expect(
            c_type.dtype == numpy.float32,
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,

            c_type.ndim >= 2,
            x_type.ndim >= 2,
            c_type.ndim == x_type.ndim,
            
            w_type.ndim == 2,

            x_type.shape[0] == c_type.shape[0],
            x_type.shape[1] == 4 * c_type.shape[1],
            
            w_type.shape[1] == 3 * c_type.shape[1],
        )
        if n_in.eval() == 4:
            b_type = in_types[3]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 2,
                b_type.shape[1] == w_type.shape[1],
            )
        
        for i in range(2, c_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == c_type.shape[i])

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x, W = inputs[0:3]
        
        if len(inputs) == 4:
            b = inputs[3]
        else:
            b = xp.zeros_like(W)
        
        a, i, f, o = _extract_gates(x)
                
        """
        i[t] = sigmoid( Wxi * x[t] + Whi * h[t-1] + Wci * c[t-1] + bi )
        f[t] = sigmoid( Wxf * x[t] + Whf * h[t-1] + Wcf * c[t-1] + bf )
        c[t] = f[t] * c[t-1] + i[t] * tanh( Wxc * x[t] + Whc * h[t-1] + bc )
        o[t] = sigmoid( Wxo * x[t] + Who * h[t-1] + Wco * c[t] + bo )
        h[t] = o[t] * tanh(c[t])
        """
        
        Wci, Wcf, Wco = _extract_gates_ph(W)
        
        bci, bcf, bco = _extract_gates_ph(b) 

        if isinstance(x, numpy.ndarray):
            self.a = numpy.tanh(a)
            self.i = _sigmoid(i + Wci * c_prev + bci)
            self.f = _sigmoid(f + Wcf * c_prev + bcf)
            self.c = self.a * self.i + self.f * c_prev
            self.o = _sigmoid(o + Wco * self.c + bco)
            h = self.o * numpy.tanh(self.c)
        else: 
            self.c, h = cuda.elementwise(
                'T c_prev, T a, T i_, T f, T o, T Wci, T Wcf, T Wco, T bci, T bcf, T bco', 
                'T c, T h',
                '''
                    T ai = sigmoid(i_ + Wci * c_prev + bci);
                    T af = sigmoid(f + Wcf * c_prev + bcf);
                    T aa = tanh(a);
                    c    = aa * ai + af * c_prev;
                    T ao = sigmoid(o + Wco * c + bco);
                    h    = ao * tanh(c);
                 ''',
                 'lstm_peephole_fwd', preamble=_preamble)(
                 c_prev, a, i, f, o, Wci, Wcf, Wco, bci, bcf, bco) 
        return self.c, h

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x, W = inputs[0:3]
        if len(inputs) == 4:
            b = inputs[3]
        else:
            b = xp.zeros_like(W)
            
        a, i, f, o = _extract_gates(x)
        gc, gh = grad_outputs
        """
        gc, gh, gW = grad_outputs[0:3]
        if len(grad_outputs) == 4:
            gb = grad_outputs[3]
        else:
            gb = xp.zeros_like(gW)
        """
        gW = xp.zeros_like(W)
        gb = xp.zeros_like(b)
        
        self.gx = xp.zeros_like(x) #xp.zeros(x.shape, dtype=numpy.float32)
        ga, gi, gf, go = _extract_gates(self.gx)
        
        Wci, Wcf, Wco = _extract_gates_ph(W)
        gWci, gWcf, gWco = _extract_gates_ph(gW)
        
        bci, bcf, bco = _extract_gates_ph(b)
        gbci, gbcf, gbco = _extract_gates_ph(gb)
        
        """
            gh[t] = sum(Wj * gy_j)                                      [0]
            go[t] = sigmoid'(xo[t]) * gh[t] * tanh(c[t])                [1]
            gc[t] = gh[t] * o[t] * tanh'(c[t]) 
                  + gc[t+1]
                  + go[t]   * Woc                                       [2.1]
            ga[t] = gc[t] * i[t] * tanh'(a[t])    where a = xc[t]       [3]
            gi[t] = gc[t] * tanh(a[t]) * sigmoid'(xi[t])                [4]
            gf[t] = gc[t] * c[t-1] * sigmoid'(xf[t])                    [5]
            gc[t] *= f[t]                                               [2.2]
            gc[t] += gi[t] * Wic + gf[t] * Wfc                          [2.3]
            
            gWci += gi[t] * c[t]
            gWcf += gf[t] * c[t]
            gWco += go[t] * c[t]
        """
        
        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0
        
        if xp is numpy:
            co = numpy.tanh(self.c)
            go[:] = gh * co * _grad_sigmoid(self.o)                             # [1]
            gc_prev  = gh * self.o * _grad_tanh(co) + gc  + go * Wco            # [2.1]
            ga[:] = gc_prev * self.i * _grad_tanh(self.a)                       # [3]
            gi[:] = gc_prev * self.a * _grad_sigmoid(self.i)                    # [4]
            gf[:] = gc_prev * c_prev * _grad_sigmoid(self.f)                    # [5]
            gc_prev *= self.f                           # multiply f here       # [2.2]
            gc_prev += gi * Wci + gf * Wcf              # add these here        # [2.3]
            
            gWci[:] = (gi[:] * c_prev).sum(axis=0, keepdims=True)
            gWcf[:] = (gf[:] * c_prev).sum(axis=0, keepdims=True)
            gWco[:] = (go[:] * self.c).sum(axis=0, keepdims=True)
            
            gbci[:] = (gi[:]).sum(axis=0, keepdims=True)
            gbcf[:] = (gf[:]).sum(axis=0, keepdims=True)
            gbco[:] = (go[:]).sum(axis=0, keepdims=True)
            
        else:
            #gx_prev = xp.empty_like(x)
            #ga_prev, gi_prev, gf_prev, go_prev = _extract_gates(gx_prev)
            gc_prev = xp.empty_like(c_prev) 
            cuda.elementwise(
                'T c_prev, T c, T gc, T gh, T a, T i_, T f, T o, T Wci, T Wcf, T Wco, T bci, T bcf, T bco',
                'T gc_prev, T ga, T gi, T gf, T go',
                '''
                    T ai = sigmoid(i_ + Wci * c_prev + bci);
                    T af = sigmoid(f  + Wcf * c_prev + bcf);
                    T aa = tanh(a);
                    T co = tanh(c);
                    T ao = sigmoid(o + Wco * c + bco);
                    go = gh * co * grad_sigmoid(ao);
                    T temp = gh * ao * grad_tanh(co) + gc + go * Wco;
                    ga = temp * ai * grad_tanh(aa);
                    gi = temp * aa * grad_sigmoid(ai);
                    gf = temp * c_prev * grad_sigmoid(af);
                    gc_prev = temp * af + gi * Wci + gf * Wcf;
                ''',
                'lstm_peephole_bwd', preamble=_preamble)(
                    c_prev, self.c, gc, gh, a, i, f, o, Wci, Wcf, Wco, bci, bcf, bco,
                    gc_prev, ga, gi, gf, go)
            
            gWci[:] = (gi[:] * c_prev).sum(axis=0, keepdims=True)
            gWcf[:] = (gf[:] * c_prev).sum(axis=0, keepdims=True)
            gWco[:] = (go[:] * self.c).sum(axis=0, keepdims=True)
            
            gbci[:] = (gi[:]).sum(axis=0, keepdims=True)
            gbcf[:] = (gf[:]).sum(axis=0, keepdims=True)
            gbco[:] = (go[:]).sum(axis=0, keepdims=True)
        
        if len(inputs) == 4:
            #return gc_prev.clip(-10., 10.), self.gx.clip(-10., 10.), gW.clip(-10., 10.), gb.clip(-10., 10.)
            return gc_prev, self.gx, gW, gb,
        else:
            #return gc_prev.clip(-10., 10.), self.gx.clip(-10., 10.), gW.clip(-10., 10.)
            return gc_prev, self.gx, gW, 

def lstm_peephole(c_prev, x, W, b=None):
    """Long Short-Term Memory units with peephole connections 
    as an activation function.
    """
    if b is None:
        return LSTMPeephole()(c_prev, x, W)
    else:
        return LSTMPeephole()(c_prev, x, W, b)
