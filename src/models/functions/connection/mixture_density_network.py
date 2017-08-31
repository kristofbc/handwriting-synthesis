import chainer
import chainer.functions

from chainer.utils import type_check
from chainer import cuda
from chainer import function
import numpy as np
#from chainer import function_node

#class MixtureDensityNetworkFunction(function_node.FunctionNode):
class MixtureDensityNetworkFunction(function.Function):

    def check_type_forward(self, in_types):
        #print("check type")
        pass
    
    def forward(self, inputs):
        x, eos_input, pi_input, mu_x1_input, mu_x2_input, s_x1_input, s_x2_input, rho_input = inputs
        #self.retain_inputs(range(len(inputs))) # Retain everything for backward

        if not type_check.same_types(*inputs):
            raise ValueError("numpy and cupy must not be used together\n"
                             "type(x): {0}, type(eos_input): {1}, type(pi_input): {2}"
                             "type(mu_x1_input): {3}, type(mu_x2_input): {4}, type(s_x1_input): {5}"
                             "type(s_x2_input): {6}, type(rho_input): {7}"
                             .format(type(x), type(eos_input), type(pi_input),
                                     type(mu_x1_input), type(mu_x2_input), type(s_x1_input),
                                     type(s_x2_input), type(rho_input)))
        
        xp = cuda.get_array_module(*inputs)

        # Get MDN coeff. Eq #18 to #22
        z_eos = 1. / (1. + xp.exp(eos_input)) # F.sigmoid. NOTE: usually sigmoid is 1/(1+e^-x). Here 'x' is >0!
        z_s_x1 = xp.exp(s_x1_input)
        z_s_x2 = xp.exp(s_x2_input)
        z_rho = xp.tanh(rho_input)
        #z_pi = xp.softmax(pi_input)
        z_pi = xp.exp(pi_input)
        z_pi = z_pi / xp.sum(z_pi, 1, keepdims=True)
        z_mu_x1 = mu_x1_input
        z_mu_x2 = mu_x2_input

        # The MDN coeff are saved, because they're reused in the backward phase
        self.z_eos = z_eos
        self.z_s_x1 = z_s_x1
        self.z_s_x2 = z_s_x2
        self.z_rho = z_rho
        self.z_pi = z_pi
        self.z_mu_x1 = z_mu_x1
        self.z_mu_x2 = z_mu_x2

        # Compute the loss.
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]
        
        # Z variable. Eq. 25
        norm_x1 = x1 - z_mu_x1
        norm_x2 = x2 - z_mu_x2
        z_left = (xp.square(norm_x1)/xp.square(z_s_x1)) + (xp.square(norm_x2)/xp.square(z_s_x2))
        z_right = (2.*z_rho*norm_x1*norm_x2)/(z_s_x1*z_s_x2)
        z = z_left - z_right
        self.z = z
        
        # Normal function. Eq. 24.
        inv_ro = 1. - xp.square(z_rho)
        n_left = 2. * np.pi * z_s_x1 * z_s_x2 * xp.sqrt(inv_ro) + 1e-10 # + 1e-10 for computational stability
        n_right = xp.exp(-z / (2. * inv_ro))
        n = n_right / n_left

        # Gamma parameter (for the backward phase). Eq. 28-29
        gamma = z_pi * n
        gamma = gamma / (xp.sum(gamma, 1, keepdims=True) + 1e-10) # sum + 1e-10 for computational stability, != nan!
        self.gamma = gamma

        # Sequence loss. Eq. 26
        loss_y = z_pi * n
        loss_y = xp.sum(loss_y, 1, keepdims=True) + 1e-10 # + 1e-10 for computational stability, != nan
        #epsilon = xp.full(loss_y.shape, 1e-10, dtype=xp.float32)
        #loss_y = xp.maximum(loss_y, epsilon) # Because at the begining loss_y is exactly 0 sometime
        loss_y = -xp.log(loss_y) 

        #loss_x = z_eos * x3 + (1. - z_eos) * (1. - x3)
        #loss_x = -xp.log(loss_x)
        loss_x = -x3 * xp.log(z_eos) - (1. - x3) * xp.log(1. - z_eos)

        loss = loss_y + loss_x

        # Mask guard to check if x3 == 2 (added padding)
        idx_mask = xp.where(x3==2)[0]
        mask = xp.ones_like(x3)
        mask[idx_mask, 0] = 0.
        self.mask = mask
        loss *= mask

        return loss, x, z_eos, z_pi, z_mu_x1, z_mu_x2, z_s_x1, z_s_x2, z_rho,

        """ Sample from the mixture to predict the next position """
        with chainer.no_backprop_mode():
            x_pred = xp.zeros((x.shape))

            for k in xrange(len(z_pi)):
                # Select the most probable id from the mixture
                treshold_idx = 0.10 # Randomly chosen treshold
                #idx = xp.argmax(z_pi[k])
                idx = z_pi[k].argmax()
                #summation = 0
                #idx = -1
                #for i in xrange(len(z_pi[k])):
                #    summation += z_pi[k][i]
                #    if summation >= treshold_idx:
                #       idx = i
                #       break

                #if idx == -1:
                #    raise ValueError("Unable to find the maximum index")

                # Get the next x1, x2
                mean = xp.asarray([z_mu_x1[k][idx], z_mu_x2[k][idx]], dtype=xp.float32)
                covar = xp.asarray([[z_s_x1[k][idx]*z_s_x1[k][idx], z_rho[k][idx]*z_s_x1[k][idx]*z_s_x2[k][idx]], [z_rho[k][idx]*z_s_x1[k][idx]*z_s_x2[k][idx], z_s_x2[k][idx]*z_s_x2[k][idx]]], dtype=xp.float32)

                # Custom implementation of multivariate normal for CuPy
                #x_normal = np.random.multivariate_normal(mean, covar, 1)
                shape = [1] # Size of sampling
                final_shape = list(shape[:])
                final_shape.append(mean.shape[0])
                x_normal = xp.random.standard_normal(final_shape).reshape(-1, mean.shape[0])
                
                # Decompose the matrix, Singular Value Decomposition
                (u, s, v) = xp.linalg.svd(covar)

                x_normal = xp.dot(x_normal, xp.sqrt(s)[:, None] * v)
                x_normal += mean
                x_normal = xp.reshape(x_normal, tuple(final_shape))

                # Sample from the distribution the next values
                x1_next, x2_next = x_normal[0][0], x_normal[0][1]

                # Get the next eos
                treshold_eos = 0.10 # Randomly chosen treshold
                x3_next = 1 if treshold_eos < z_eos[k][0] else 0

                x_pred[k, 0] = x1_next
                x_pred[k, 1] = x2_next
                x_pred[k, 2] = x3_next

        return loss, x_pred, z_eos, z_pi, z_mu_x1, z_mu_x2, z_s_x1, z_s_x2, z_rho,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        #x, eos_input, pi_input, mu_x1_input, mu_x2_input, s_x1_input, s_x2_input, rho_input = self.get_retained_inputs()
        x, eos_input, pi_input, mu_x1_input, mu_x2_input, s_x1_input, s_x2_input, rho_input = inputs

        # MDN coeff to differentiate
        g_eos = xp.empty_like(eos_input)
        g_s_x1 = xp.empty_like(s_x1_input)
        g_s_x2 = xp.empty_like(s_x2_input)
        g_rho = xp.empty_like(rho_input)
        g_pi = xp.empty_like(pi_input)
        g_mu_x1 = xp.empty_like(mu_x1_input)
        g_mu_x2 = xp.empty_like(mu_x2_input)

        # Compute the gradient
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]

        # From eq. 27 to 37
        C = 1. / (1. - rho_input*rho_input)
        d_norm_x1 = (x1 - self.z_mu_x1) / self.z_s_x1
        d_norm_x2 = (x2 - self.z_mu_x2) / self.z_s_x2
        d_rho_norm_x1 = self.z_rho * d_norm_x1
        d_rho_norm_x2 = self.z_rho * d_norm_x2

        g_eos = x3 - self.z_eos
        g_pi = self.z_pi - self.gamma
        g_mu_x1 = (C/self.z_s_x1) * (d_norm_x1 - d_rho_norm_x2)
        g_mu_x2 = (C/self.z_s_x2) * (d_norm_x2 - d_rho_norm_x1)
        g_s_x1 = (C*d_norm_x1) * (d_norm_x1 - d_rho_norm_x2) - 1.
        g_s_x2 = (C*d_norm_x2) * (d_norm_x2 - d_rho_norm_x1) - 1.
        g_rho = d_norm_x1*d_norm_x2 + self.z_rho*(1. - C * self.z)
        
        # Add grad_clipping here if it explodes P.23

        return None, g_eos, g_pi, g_mu_x1, g_mu_x2, g_s_x1, g_s_x2, g_rho,


def mixture_density_network(x, eos, pi, mu_x1, mu_x2, s_x1, s_x2, rho):
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
    return MixtureDensityNetworkFunction()(x, eos, pi, mu_x1, mu_x2, s_x1, s_x2, rho)
