import chainer
import chainer.functions as F
from chainer import cuda

from functions.connection.mixture_density_network import mixture_density_network

class MixtureDensityNetwork(chainer.Link):
    """
        The Mixture-Density-Network outputs a parametrised mixture distribution.
        
        Args:
            n_mdn_comp (int): number of MDN components
            n_units (int): number of cells units
            prob_bias (float): bias added to the pi's probability
    """
    def __init__(self, n_mdn_comp, n_units, prob_bias = 0.):
        super(MixtureDensityNetwork, self).__init__()

        self.n_mdn_comp = n_mdn_comp
        self.n_unit = n_units
        self.p_bias = prob_bias

        self.eos, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = None, None, None, None, None, None, None
        self.gamma = None
        self.loss = None

    def reset_state(self):
        """
            Reset the Variables
        """
        self.eos, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = None, None, None, None, None, None, None
        self.gamma = None
        self.loss = None
 
    def __call__(self, inputs):
        """
            Perform the MDN prediction

            Args:
                inputs (float[][]): input tensor 
            Returns:
                loss (float)
        """
        x, y = inputs
        xp = cuda.get_array_module(*x)

        # Extract the MDN's parameters
        eos, pi_h, mu_x1_h, mu_x2_h, s_x1_h, s_x2_h, rho_h = F.split_axis(
            y, [1 + i*self.n_mdn_comp for i in xrange(5+1)], axis=1
        )

        # Add the bias to the parameter to change the shape of the prediction
        pi_h *= (1. + self.p_bias)
        s_x1_h -= self.p_bias
        s_x2_h -= self.p_bias

        self.loss, _, self.eos, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = mixture_density_network(
            x, eos, pi_h, mu_x1_h, mu_x2_h, s_x1_h, s_x2_h, rho_h
        )

        return self.loss 
