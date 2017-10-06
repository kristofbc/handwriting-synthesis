import chainer
import chainer.functions as F
from chainer import variable
from chainer import initializers

from initializers.normal_bias import NormalBias

class AdaptiveWeightNoise(chainer.Link):
    """
        Alex Grave's Adaptive Weight Noise
        From: Practical Variational Inference for Neural Networks.
    """
    def __init__(self, in_size, out_size, normal_scale=0.1, nobias=False, initial_bias=0):
        super(AdaptiveWeightNoise, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.normal_scale = normal_scale
        self.nobias = nobias
        self.initial_bias = initial_bias

        with self.init_scope():
            if nobias:
                self.mu = chainer.Parameter(chainer.initializers.Normal(1))
            else:
                self.mu = chainer.Parameter(NormalBias(self.out_size, self.normal_scale, self.initial_bias))

            self.sigma = chainer.Parameter(initializers._get_initializer(self.xp.log(1e-10)))

            if in_size is not None:
                self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.mu.initialize((self.out_size*in_size))
        self.sigma.initialize((self.out_size*in_size))
        self.in_size = in_size

    def get_test_weight(self, in_size=None):
        """
            When testing, do not generate AWN weights and biases, use the current configuration
            Args:
                in_size (int): Input size of the variable to transform
            Returns:
                weights(float[][])
        """
        if self.in_size is None:
            if in_size is None:
                raise ValueError("in_size should not be none for test weights")

            self.in_size = in_size

        if self.nobias:
            return F.reshape(self.mu, (self.out_size, self.in_size)), None
        else:
            w, b = F.split_axis(self.mu, [self.out_size*(self.in_size-1)], axis=0)
            return F.reshape(w, (self.out_size, self.in_size-1)), b

    def __call__(self, batch_size, in_size=None):
        """
            Update the weigths
            Args:
                batch_size (Variable): Size of the current batch
                in_size (int): Input size of the variable to transform
            Returns:
                weight (float[][]), loss (float)
        """
        if self.mu.data is None or self.sigma.data is None:
            if in_size is None:
                raise ValueError("AdaptiveWeightNoise requires a in_size to intialize it's Parameters")

            self._initialize_params(in_size)

        # Base parameters
        mu_h = F.broadcast_to(F.mean(self.mu), self.mu.shape)
        diff_mu_h = F.square(self.mu - mu_h)
        sigma_h = F.broadcast_to(F.mean(F.exp(self.sigma) + diff_mu_h), self.sigma.shape)

        # Weight and bias
        eps = variable.Variable(self.xp.random.randn(self.mu.size).astype(self.xp.float32))
        W = self.mu + eps * F.sqrt(F.exp(self.sigma))

        # Loss
        loss_x = (F.log(sigma_h) - self.sigma) / 2.
        loss_y = (diff_mu_h + F.exp(self.sigma) - sigma_h) / ((2. * sigma_h) + 1e-8)
        loss = F.reshape(F.sum(loss_x + loss_y), (1,)) / batch_size

        # Extract the bias if required
        if self.nobias:
            return F.reshape(W, (self.out_size, self.in_size)), None, loss
        else:
            w, b = F.split_axis(W, self.xp.asarray([self.out_size*(self.in_size-1)]), axis=0)
            return F.reshape(w, (self.out_size, self.in_size-1)), b, loss

