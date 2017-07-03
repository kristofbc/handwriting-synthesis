import numpy

from chainer.functions.connection import linear
from chainer import link
#from chainer.links.connection import linear
from chainer import variable
from chainer import cuda
from chainer import function
from chainer.utils import type_check

def _extract_gates(x):
    r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
    return (r[:, :, i] for i in six.moves.range(4))


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
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''

class LSTM(link.Link):
    """Long short-term memory unit with forget gate.

    It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
    state. x must have four times channels compared to the number of units.

    """
    def __init__(self, out_size):
        super(LSTM, self).__init__()
        self.state_size = out_size
        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        c_type, x_type = in_types

        type_check.expect(
            c_type.dtype == numpy.float32,
            x_type.dtype == numpy.float32,

            c_type.ndim >= 2,
            x_type.ndim >= 2,
            c_type.ndim == x_type.ndim,

            x_type.shape[0] == c_type.shape[0],
            x_type.shape[1] == 4 * c_type.shape[1],
        )
        for i in range(2, c_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == c_type.shape[i])

    def forward(self, inputs):
        c_prev, x = inputs
        a, i, f, o = _extract_gates(x)

        if isinstance(x, numpy.ndarray):
            self.a = numpy.tanh(a)
            self.i = _sigmoid(i)
            self.f = _sigmoid(f)
            self.o = _sigmoid(o)

            self.c = self.a * self.i + self.f * c_prev
            h = self.o * numpy.tanh(self.c)
        else:
            self.c, h = cuda.elementwise(
                'T c_prev, T a, T i_, T f, T o', 'T c, T h',
                '''
                    COMMON_ROUTINE;
                    c = aa * ai + af * c_prev;
                    h = ao * tanh(c);
                ''',
                'lstm_fwd', preamble=_preamble)(c_prev, a, i, f, o)

        return self.c, h

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x = inputs
        gc, gh = grad_outputs

        gx = xp.empty_like(x)
        ga, gi, gf, go = _extract_gates(gx)
        
        batchsize = x.shape[0]        

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        if xp is numpy:
            co = numpy.tanh(self.c)
            gc_prev = gh * self.o * _grad_tanh(co) + gc  # multiply f later
            ga[:] = gc_prev * self.i * _grad_tanh(self.a)
            gi[:] = gc_prev * self.a * _grad_sigmoid(self.i)
            gf[:] = gc_prev * c_prev * _grad_sigmoid(self.f)
            go[:] = gh * co * _grad_sigmoid(self.o)
            gc_prev *= self.f  # multiply f here
        else:
            a, i, f, o = _extract_gates(x)
            gc_prev = xp.empty_like(c_prev)
            cuda.elementwise(
                'T c_prev, T c, T gc, T gh, T a, T i_, T f, T o',
                'T gc_prev, T ga, T gi, T gf, T go',
                '''
                    COMMON_ROUTINE;
                    T co = tanh(c);
                    T temp = gh * ao * grad_tanh(co) + gc;
                    ga = temp * ai * grad_tanh(aa);
                    gi = temp * aa * grad_sigmoid(ai);
                    gf = temp * c_prev * grad_sigmoid(af);
                    go = gh * co * grad_sigmoid(ao);
                    gc_prev = temp * af;
                ''',
                'lstm_bwd', preamble=_preamble)(
                    c_prev, self.c, gc, gh, a, i, f, o,
                    gc_prev, ga, gi, gf, go)
        
        gc_prev_max = xp.max(xp.absolute(gc_prev), axis=1).reshape((batchsize, 1))
        rate = xp.where(gc_prev_max > 10.0, 10.0/gc_prev_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        gc_prev *= rate
        
        gx_max = xp.max(xp.absolute(gx), axis=1).reshape((batchsize, 1))
        rate = xp.where(gx_max > 10.0, 10.0/gx_max, 1.0).astype(xp.float32).reshape((batchsize, 1))
        gx *= rate

        #return gc_prev.clip(-10.0, 10.0), gx.clip(-10.0, 10.0)
        return gc_prev, gx

    def __call__(self, x, W_lateral, b_lateral):
        
        lstm_in = x
        if self.h is not None:
            lstm_in += linear.linear(self.h, W_lateral, b_lateral)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        self.c, self.h = super(LSTM, self).__call__(self.c, lstm_in)
        return self.h

