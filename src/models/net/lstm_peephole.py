import numpy

from net.functions.lstm_peephole import lstm_peephole
from chainer.functions.connection import linear
from chainer import link
#from chainer.links.connection import linear
from chainer import variable


class LSTMPeephole(link.Chain):
    
    def __init__(self, out_size):
        self.state_size = out_size
        self.reset_state()
        

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, x, W_lateral, W_peephole, b_peephole):
        lstm_in = x
        if self.h is not None:
            lstm_in += linear.linear(self.h, W_lateral)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        self.c, self.h = lstm_peephole(self.c, lstm_in, W_peephole, b_peephole)
        return self.h
        
        
        
