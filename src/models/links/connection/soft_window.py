import chainer
import chainer.functions as F
from chainer import variable

from functions.connection.soft_window import SoftWindowFunction

class SoftWindow(chainer.Link):
    """
        The SoftWindow act as an attention mechanism which controls the alignment between the text and the pen position

        Args:
            mixture_size (int): size of the mixture "k"
            unit_size (int): size of the mixture hidden units
    """
    def __init__(self, mixture_size, unit_size):
        super(SoftWindow, self).__init__()

        self.mixture_size = mixture_size
        self.unit_size = unit_size
        self.k_prev = None
        self.w = None
        self.phi = None

    def reset_state(self):
        """
            Reset Variables
        """
        self.k_prev = None
        self.w = None
        self.phi = None
    
    def __call__(self, inputs):
        """
            Perform the SoftWindow text-pen alignment prediction

            Args:
                inputs (float[][]): input tensor containing "x" and the character sequence "cs"
            Returns:
                window (float)
        """
        x, cs = inputs
        batch_size, W, u = cs.shape

        # Extract the soft window's parameters
        a_h, b_h, k_h = F.split_axis(x, [self.mixture_size, 2 * self.mixture_size], axis=1)
        K = a_h.shape[1]

        if self.k_prev is None:
            self.k_prev = variable.Variable(self.xp.zeros_like(k_h, dtype=self.xp.float32))

        self.w, self.k_prev, self.phi = SoftWindowFunction()(a_h, b_h, k_h, cs, self.k_prev)

        return self.w
