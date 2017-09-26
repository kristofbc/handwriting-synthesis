import chainer
import chainer.functions

from chainer.utils import type_check
from chainer import cuda
from chainer import function
import chainer.functions as F
import numpy as np
from chainer.functions.activation import log_softmax
#from chainer import function_node

from utils import clip_grad

#class MixtureDensityNetworkFunction(function_node.FunctionNode):
class MixtureDensityNetworkFunction(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 8)
        x_type, q_input_type, pi_input_type, mu_x1_input_type, mu_x2_input_type, s_x1_input_type, s_x2_input_type, rho_input_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            q_input_type.dtype.kind == 'f',
            pi_input_type.dtype.kind == 'f',
            mu_x1_input_type.dtype.kind == 'f',
            mu_x2_input_type.dtype.kind == 'f',
            s_x1_input_type.dtype.kind == 'f',
            s_x2_input_type.dtype.kind == 'f',
            rho_input_type.dtype.kind == 'f',

            x_type.ndim >= 2,
            q_input_type.ndim >= 2,

            x_type.shape[0] == q_input_type.shape[0],
            x_type.shape[0] == pi_input_type.shape[0],
            x_type.shape[0] == mu_x1_input_type.shape[0],
            x_type.shape[0] == mu_x2_input_type.shape[0],
            x_type.shape[0] == s_x1_input_type.shape[0],
            x_type.shape[0] == s_x2_input_type.shape[0],
            x_type.shape[0] == rho_input_type.shape[0],

            pi_input_type.shape[1] == mu_x1_input_type.shape[1],
            mu_x1_input_type.shape[1] == mu_x2_input_type.shape[1],
            mu_x2_input_type.shape[1] == s_x1_input_type.shape[1],
            s_x1_input_type.shape[1] == s_x2_input_type.shape[1],
            s_x2_input_type.shape[1] == rho_input_type.shape[1]
        )

        pass
    
    def forward(self, inputs):
        x, q_input, pi_input, mu_x1_input, mu_x2_input, s_x1_input, s_x2_input, rho_input = inputs
        #self.retain_inputs(range(len(inputs))) # Retain everything for backward

        if not type_check.same_types(*inputs):
            raise ValueError("numpy and cupy must not be used together\n"
                             "type(x): {0}, type(q_input): {1}, type(pi_input): {2}"
                             "type(mu_x1_input): {3}, type(mu_x2_input): {4}, type(s_x1_input): {5}"
                             "type(s_x2_input): {6}, type(rho_input): {7}"
                             .format(type(x), type(q_input), type(pi_input),
                                     type(mu_x1_input), type(mu_x2_input), type(s_x1_input),
                                     type(s_x2_input), type(rho_input)))
        
        xp = cuda.get_array_module(*inputs)

        epsilon = 1e-10
        def softmax(x):
            shiftx = x - x.max()
            exps = xp.exp(shiftx)
            return exps / xp.sum(exps, 1, keepdims=True)

        # Get MDN coeff. Eq #18 to #22
        #z_q = log_softmax._log_softmax(q_input)
        z_q = softmax(q_input)
        z_s_x1 = xp.exp(s_x1_input) + epsilon
        z_s_x2 = xp.exp(s_x2_input) + epsilon
        z_rho = xp.tanh(rho_input)
        z_pi = softmax(pi_input)
        #z_pi = xp.exp(pi_input)
        #z_pi = z_pi / xp.sum(z_pi, 1, keepdims=True)
        z_mu_x1 = mu_x1_input
        z_mu_x2 = mu_x2_input

        # The MDN coeff are saved, because they're reused in the backward phase
        #self.z_q = xp.exp(z_q)
        self.z_q = z_q
        self.z_s_x1 = z_s_x1
        self.z_s_x2 = z_s_x2
        self.z_rho = z_rho
        self.z_pi = z_pi
        self.z_mu_x1 = z_mu_x1
        self.z_mu_x2 = z_mu_x2

        # Compute the loss.
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3_5 = x[:, 2:5]
        
        # Z variable. Eq. 25
        norm_x1 = x1 - z_mu_x1
        norm_x2 = x2 - z_mu_x2
        z_left = (xp.square(norm_x1)/xp.square(z_s_x1)) + (xp.square(norm_x2)/xp.square(z_s_x2))
        z_right = (2.*z_rho*norm_x1*norm_x2)/(z_s_x1*z_s_x2)
        z = z_left - z_right
        self.z = z
        
        # Normal function. Eq. 24.
        inv_ro = 1. - xp.square(z_rho) + epsilon
        n_left = 2. * np.pi * z_s_x1 * z_s_x2 * xp.sqrt(inv_ro) + epsilon # + 1e-10 for computational stability
        n_right = xp.exp(-z / (2. * inv_ro))
        n = n_right / n_left

        # Gamma parameter (for the backward phase). Eq. 28-29
        gamma = z_pi * n
        gamma = gamma / (xp.sum(gamma, 1, keepdims=True) + epsilon) # sum + 1e-10 for computational stability, != nan!
        self.gamma = gamma

        # Sequence loss. Eq. 26
        loss_y = z_pi * n
        loss_y = xp.sum(loss_y, 1, keepdims=True) + epsilon # + 1e-10 for computational stability, != nan
        #epsilon = xp.full(loss_y.shape, 1e-10, dtype=xp.float32)
        #loss_y = xp.maximum(loss_y, epsilon) # Because at the begining loss_y is exactly 0 sometime
        loss_y = -xp.log(loss_y + epsilon) 
        
        # Softmax cross-entropy
        loss_x = -x3_5 * xp.log(z_q + epsilon)
        self.loss_q = loss_x
        #loss_x = xp.reshape(loss_x, (-1, 1))

        loss = loss_y + loss_x

        # Mask guard to check if x3 == 2 (added padding)
        idx_mask = xp.where(x3_5[:, 0] == 2)[0]
        mask = xp.ones((len(x3_5), 1), dtype=xp.float32) # Only 1D array for mask
        mask[idx_mask, 0] = 0.
        self.mask = mask
        loss *= mask

        return loss, loss_y, loss_x, x, z_q, z_pi, z_mu_x1, z_mu_x2, z_s_x1, z_s_x2, z_rho,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, q_input, pi_input, mu_x1_input, mu_x2_input, s_x1_input, s_x2_input, rho_input = inputs

        epsilon = 1e-10

        # MDN coeff to differentiate
        g_q = xp.empty_like(q_input)
        g_s_x1 = xp.empty_like(s_x1_input)
        g_s_x2 = xp.empty_like(s_x2_input)
        g_rho = xp.empty_like(rho_input)
        g_pi = xp.empty_like(pi_input)
        g_mu_x1 = xp.empty_like(mu_x1_input)
        g_mu_x2 = xp.empty_like(mu_x2_input)

        # Compute the gradient
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3_5 = x[:, 2:5]

        #if xp == np:
            # From eq. 27 to 37
        C = 1. / (1. - self.z_rho*self.z_rho + epsilon)
        d_norm_x1 = (x1 - self.z_mu_x1) / self.z_s_x1
        d_norm_x2 = (x2 - self.z_mu_x2) / self.z_s_x2
        d_rho_norm_x1 = self.z_rho * d_norm_x1
        d_rho_norm_x2 = self.z_rho * d_norm_x2

        #g_q = (xp.log(self.z_q + epsilon) - 1) * self.loss_q
        g_q = (self.z_q - 1) * self.mask
        g_pi = (self.z_pi - self.gamma) * self.mask
        g_mu_x1 = - self.gamma * ((C/self.z_s_x1) * (d_norm_x1 - d_rho_norm_x2)) * self.mask
        g_mu_x2 = - self.gamma * ((C/self.z_s_x2) * (d_norm_x2 - d_rho_norm_x1)) * self.mask
        g_s_x1 = - self.gamma * ((C*d_norm_x1) * (d_norm_x1 - d_rho_norm_x2) - 1.) * self.mask
        g_s_x2 = - self.gamma * ((C*d_norm_x2) * (d_norm_x2 - d_rho_norm_x1) - 1.) * self.mask
        g_rho = - self.gamma * (d_norm_x1*d_norm_x2 + self.z_rho*(1. - C * self.z)) * self.mask
        
        #else:
        #    g_eos, g_pi, g_mu_x1, g_mu_x2, g_s_x1, g_s_x2, g_rho = cuda.elementwise(
        #        'T x1, T x2, T eos_input, T pi_input, T mu_x1_input, T mu_x2_input, T s_x1_input, T s_x2_input, T rho_input',
        #        'T g_eos, T g_pi, T g_mu_x1, T g_mu_x2, T g_s_x1, T g_s_x2, T g_rho',

        #    )
        
        # Add grad_clipping here if it explodes P.23
        th_min = -100.0
        th_max = 100.0

        g_q = clip_grad(g_q, th_min, th_max, xp)
        g_pi = clip_grad(g_pi, th_min, th_max, xp)
        g_mu_x1 = clip_grad(g_mu_x1, th_min, th_max, xp)
        g_mu_x2 = clip_grad(g_mu_x2, th_min, th_max, xp)
        g_s_x1 = clip_grad(g_s_x1, th_min, th_max, xp)
        g_s_x2 = clip_grad(g_s_x2, th_min, th_max, xp)
        g_rho = clip_grad(g_rho, th_min, th_max, xp)

        return None, g_q, g_pi, g_mu_x1, g_mu_x2, g_s_x1, g_s_x2, g_rho,


def mixture_density_network(x, q, pi, mu_x1, mu_x2, s_x1, s_x2, rho):
    """ Mixture Density Network
        
        Output the coefficient params 

        Args:
            x (Variable): Tensor containing the position [x1, x2, x3] to predict
            eos (Variable): End-of-stroke prediction
            pi (Variable): mixture components
            mu_x1 (Variable): mean of x1
            mu_x2 (Variable): mean of x2
            s_x1 (Variable): variance of x1
            s_x2 (Variable): variance of x2
            rho (Variable): correlation parameter

        Returns:
            loss (Variable)
            y (Variable)
            eos (Variable)
            pi (Variable)
            mu_x1 (Variable)
            mu_x2 (Variable)
            s_x1 (Variable)
            s_x2 (Variable)
            rho (Variable)
    """
    return MixtureDensityNetworkFunction()(x, q, pi, mu_x1, mu_x2, s_x1, s_x2, rho)
