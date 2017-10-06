import chainer
from chainer import function
from chainer import cuda
from chainer.utils import type_check

class SoftWindowFunction(function.Function):
    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        a_h, b_h, k_h, cs, k_prev = inputs

        if not type_check.same_types(*inputs):
            raise ValueError("numpy and cupy must not be used together\n"
                             "type(a_h): {0}, type(b_h): {1}, type(k_h): {2}, type(cs): {3}, type(k_prev): {4}"
                             .format(type(a_h), type(b_h), type(k_h), type(cs), type(k_prev)))

        batch_size, W, u = cs.shape
        K = a_h.shape[1]
        xp = cuda.get_array_module(*inputs)

        a_h = xp.exp(a_h).reshape((batch_size, K, 1))
        b_h = xp.exp(b_h).reshape((batch_size, K, 1))
        k_h = k_prev + xp.exp(k_h)
        k_h = xp.reshape(k_h, (batch_size, K, 1))

        self.a_h = a_h
        self.b_h = b_h
        self.k_h = k_h

        # Compute phi's parameters
        #us = xp.linspace(0, u-1, u)
        us = xp.arange(u, dtype=xp.float32).reshape((1, 1, u))
        phi = a_h * xp.exp(-b_h * xp.square(k_h - us))

        self.phi = phi
        
        phi = xp.sum(phi, axis=1)

        # Finalize the soft window computation
        w = xp.matmul(cs, phi.reshape((batch_size, u, 1)))

        return w.reshape((batch_size, W)), k_h.reshape((batch_size, K)), phi

    def backward(self, inputs, grad_outputs):
        a_h, b_h, k_h, cs, k_prev = inputs

        batch_size, W, u = cs.shape
        K = a_h.shape[1]
        xp = cuda.get_array_module(*inputs)
        us = xp.arange(u, dtype=xp.float32).reshape((1, 1, u))

        # Get the gradient output w.r.t the loss fow "w" and "k"
        gw, gk = grad_outputs[0:2]

        if gw is None:
            gw = 0.
        if gk is None:
            gk = 0.

        # Compute e using forward and gradient values. Eq. 56
        g_e = self.phi * xp.matmul(gw.reshape(batch_size, 1, W), cs)

        # Gradients of the original parameters. Eq. 57 to 60
        g_a_h = xp.sum(g_e, axis=2)
        b = self.b_h.reshape((batch_size, K))
        g_b_h = -b * xp.sum(g_e * xp.square(self.k_h - us), axis=2)
        g_k_p = gk + 2. * b * xp.sum(g_e * (us - self.k_h), axis=2) 
        g_k_h = xp.exp(k_h) * g_k_p

        return g_a_h, g_b_h, g_k_h, None, g_k_p,


def soft_window(a_h, b_h, k_h, cs, k_prev):
    """
        Perform the SoftWindow call
    """
    return SoftWindowFunction()(a_h, b_h, k_h, cs, k_prev)

