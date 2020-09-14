import torch
import torch.nn as nn
from pyro.nn import ConditionalAutoRegressiveNN


class BatchNormFlow(torch.nn.Module):
    """Implements the batch normalization flow described in MAF paper,
    1705.07057, appendix B.

    Code adapted from pyro and
    https://github.com/kamenbliznashki/normalizing_flows

    The flow takes

    u -> x = f(u)
           = (u - beta) * exp(-gamma) * (v + epsilon)^{1/2} + m

    where gamma and beta are learned parameters, and m and v are the
    mean and variance calculated over the mini-batch during training.

    At test time, m and v are the running mean and variance, or they
    can be taken to be averages over the training set."""

    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        super(BatchNormFlow, self).__init__()

        self.input_size = input_size
        self.gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))
        self.momentum = momentum
        self.epsilon = epsilon
        self.averaging = False
        self.count = 0

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x):
        """This is really the inverse transformation, x -> f^{-1}(x) = u, as
        given in equation (22) of MAF paper.

        Returns:

            u = f^{-1}(x)

            log det dx/du, a scalar

        """
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)

            # update running mean and var
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(
                self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var

        elif self.averaging:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)

            self.running_mean.mul_(self.count).add_(
                self.batch_mean.data).mul_(1.0 / (self.count + 1))
            self.running_var.mul_(self.count).add_(
                self.batch_var.data).mul_(1.0 / (self.count + 1))
            self.count += 1

            mean = self.batch_mean
            var = self.batch_var

        else:
            mean = self.running_mean
            var = self.running_var

        # Normalized input
        u = ((x - mean) / torch.sqrt(var + self.epsilon)) * \
            self.gamma.exp() + self.beta

        # log|det J(f)|
        # This is minus the log of equation (23)
        log_abs_det_jacobian = - \
            (self.gamma - 0.5 * torch.log(var + self.epsilon)).sum()

        return u, log_abs_det_jacobian

    def inverse(self, u):
        """x(u), given by equation (21)"""
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (u - self.beta) * torch.exp(-self.gamma) * torch.sqrt(
            var + self.epsilon
        ) + mean
        log_abs_det_jacobian = - \
            (self.gamma - 0.5 * torch.log(var + self.epsilon)).sum()

        return x, log_abs_det_jacobian

    def start_averaging(self):
        self.averaging = True
        self.running_mean.zero_()
        self.running_var.zero_()
        self.count = 0

    def end_averaging(self):
        self.averaging = False


class MAF(torch.nn.Module):
    """Class containing a single MAF block. It can be used also as an IAF
    block.

    The forward pass is always the fast direction. For a MAF, this means that
    it actually computes the inverse transformation f^{-1}(x). The inverse pass
    is the slow pass (i.e., for the MAF, f(u)).

    There are two parametrizations that are implemented:

        standard:       x = f(u) = u * exp(alpha(x, y)) + mu(alpha(x, y))

                where alpha(x, y) and mu(x, y) are autoregressive on x, and y
                is a context variable. I.e., there can be arbitrary dependence
                on y.

        iaf:            u = f^{-1}(x) = x * sigma(x, y)
                                        + (1 - sigma(x, y)) * m(x, y)

                where sigma(x, y) = sigmoid(s(x, y))

                and where s(x, y) and m(x, y) are autoregressive on x, and y
                is a context variable.

    The forward pass, therefore, returns f^{-1}(x), and the inverse pass
    returns f(u). In addition, both passes return log|det J(f)|, where J(f) is
    the Jacobian of f(u).

    Args:

            input_dim       dim(x)
            context_dim     dim(y)
            hidden_dims     list of dimensions for hidden layers of
                            autoregressive network (MADE)
            nonlinearity    activation function for autoregressive network
            iaf_parametrization     whether to use iaf parametrization. If
                                    False, uses standard parametrization.
    """

    def __init__(self, input_dim, context_dim, hidden_dims,
                 activation=nn.ELU(), iaf_parametrization=False):
        super(MAF, self).__init__()
        self.arn = ConditionalAutoRegressiveNN(
            input_dim,
            context_dim,
            hidden_dims,
            nonlinearity=activation,
            skip_connections=False,
        )

        # This is a bit of a hack: it registers the perturbation attribute of
        # the ConditionalAutoregressiveNN as a buffer, so that it will be saved
        # when state_dict() is called. We call permutation explicitly below, so
        # this quantity needs to be saved.
        #
        # A less hacky approach would be to save each permutation as a
        # hyperparameter and then pass at build time. This is quicker though.

        perm = self.arn.permutation.clone().detach()
        del self.arn.permutation
        self.arn.register_buffer('permutation', perm)

        self.iaf_parametrization = iaf_parametrization
        self.initial_bias = 1.0

    def inverse(self, u, context):
        """This is the forward MAF."""

        x_size = u.size()[:-1]
        perm = self.arn.permutation
        input_size = u.size(-1)
        x = [torch.zeros(x_size, device=u.device)] * input_size

        # Expensive
        for idx in perm:
            if self.iaf_parametrization:
                m, s = self.arn(torch.stack(x, dim=-1), context)
                sigma = torch.sigmoid(
                    s + self.initial_bias * torch.ones_like(s))
                x[idx] = ((u[..., idx] - m[..., idx]) / sigma[..., idx] +
                          m[..., idx])
            else:
                mu, alpha = self.arn(torch.stack(x, dim=-1), context)
                x[idx] = (u[..., idx] * torch.exp(alpha[..., idx]) +
                          mu[..., idx])
        x = torch.stack(x, dim=-1)

        # log|det df/du|
        if self.iaf_parametrization:
            log_det_Jf = - (torch.log(sigma)).sum(-1)
        else:
            log_det_Jf = alpha.sum(-1)

        return x, log_det_Jf

    def forward(self, x, context):
        """This is really the inverse pass of the flow."""
        if self.iaf_parametrization:
            m, s = self.arn(x, context)
            # Initial bias of s to improve training. Initially, the IAF
            # does not effect a large change in x.
            sigma = torch.sigmoid(s + self.initial_bias * torch.ones_like(s))
            u = x * sigma + (torch.ones_like(sigma) - sigma) * m
            log_det_Jf = - (torch.log(sigma)).sum(-1)
        else:
            mu, alpha = self.arn(x, context)
            u = (x - mu) * torch.exp(-alpha)
            log_det_Jf = alpha.sum(-1)

        # Return the result of the inverse flow, along with log|det J(f)|
        return u, log_det_Jf


class MAFStack(torch.nn.Module):
    """Container of a stack of MAF blocks.

    """

    def __init__(self,
                 input_dim,
                 context_dim,
                 hidden_dims,
                 nflows,
                 batch_norm=True,
                 bn_momentum=0.9,
                 activation='elu',
                 iaf_parametrization=False
                 ):
        super(MAFStack, self).__init__()

        if activation == 'elu':
            activation_fn = nn.ELU()
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            activation_fn = nn.LeakyReLU()
        else:
            activation_fn = nn.ReLU()   # Default
            print('Invalid activation function specified. Using ReLU.')

        self.maf_list = nn.ModuleList(
            [MAF(input_dim, context_dim, hidden_dims, activation=activation_fn,
                 iaf_parametrization=iaf_parametrization)
             for i in range(nflows)])

        if batch_norm:
            self.bnf_list = nn.ModuleList(
                [BatchNormFlow(input_dim, momentum=bn_momentum)
                 for i in range(nflows)])

        self.batch_norm = batch_norm

        self.model_hyperparams = {
            'input_dim': input_dim,
            'context_dim': context_dim,
            'hidden_dims': hidden_dims,
            'nflows': nflows,
            'batch_norm': batch_norm,
            'bn_momentum': bn_momentum,
            'activation': activation,
            'iaf_parametrization': iaf_parametrization
        }

    def inverse(self, u, context):
        x = u
        log_det_total = x.new_zeros(u.size()[:-1])
        for i, maf in enumerate(self.maf_list):
            if self.batch_norm:
                bnf = self.bnf_list[i]
                x, log_det = bnf.inverse(x)
                log_det_total = log_det_total + log_det
            x, log_det = maf.inverse(x, context)
            log_det_total = log_det_total + log_det
        return x, log_det_total

    def forward(self, x, context):
        u = x
        log_det_total = x.new_zeros(x.size()[:-1])
        for i, maf in reversed(list(enumerate(self.maf_list))):
            u, log_det = maf(u, context)
            log_det_total = log_det_total + log_det
            if self.batch_norm:
                bnf = self.bnf_list[i]
                u, log_det = bnf(u)
                log_det_total = log_det_total + log_det
        return u, log_det_total

    def start_averaging(self):
        for i, bnf in enumerate(self.bnf_list):
            bnf.start_averaging()

    def end_averaging(self):
        for i, bnf in enumerate(self.bnf_list):
            bnf.end_averaging()


def train_epoch(flow, base_dist, train_loader, optimizer, epoch,
                device=None,
                output_freq=50):
    """Train MAF flow for 1 epoch.

    Args:
        flow:           instance of MAFStack
        base_dist:      base distribution from which to flow
        train_loader:   instance of pytorch DataLoader
        optimizer:      optimizer instance
        epoch:          int, epoch number
        device:         device to use
        output_freq:    number of batches between printed outputs
                            default 50
                            None suppresses outputs

    Returns:
        average loss over epoch
    """
    flow.train()
    train_loss = 0.0

    for batch_idx, (h, x) in enumerate(train_loader):
        optimizer.zero_grad()

        if device is not None:
            h = h.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)

        # Sample a noise realization
        # y_dist = torch.distributions.Normal(loc=h, scale=torch.ones_like(h))
        # y = y_dist.sample()

        y = h + torch.randn_like(h)

        # Flow to base distribution
        u, neg_log_prob = flow(x, context=y)

        # Loss is negative log prob
        loss = - base_dist.log_prob(u) + neg_log_prob

        # Keep track of total loss
        train_loss += loss.sum()

        loss = loss.mean()

        loss.backward()
        optimizer.step()

        if (output_freq is not None) and (batch_idx % output_freq == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx *
                train_loader.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    train_loss = train_loss.item() / len(train_loader.dataset)
    print('Train Epoch: {} \tAverage Loss: {:.4f}'.format(
        epoch, train_loss))

    return train_loss


def test_epoch(flow, base_dist, test_loader, device=None):
    """Calculate validation loss.

    Args:
        flow:           instance of MAFStack
        base_dist:      base distribution from which to flow
        test_loader:   instance of pytorch DataLoader
        device:         device to use

    Returns:
        average loss over test_loader
    """
    with torch.no_grad():
        flow.eval()
        test_loss = 0.0
        for h, x in test_loader:

            if device is not None:
                h = h.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)

            # Sample a noise realization
            # y_dist = torch.distributions.Normal(loc=h,
            #                                    scale=torch.ones_like(h))
            # y = y_dist.sample()

            y = h + torch.randn_like(h)

            # Flow
            u, neg_log_prob = flow(x, y)

            # Loss is negative log prob
            loss = - base_dist.log_prob(u) + neg_log_prob

            # Keep track of total loss
            test_loss += loss.sum()

        test_loss = test_loss.item() / len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}\n'
              .format(test_loss))

        return test_loss


def recalculate_moving_avgs(flow, train_loader, device):
    """Reset the moving averages of the batch norm layers contained
    within the MAF flow to be the averages over the entire training set.

    This is necessary in order to achieve good performance on test set.

    Args:
        flow:           instance of MAFStack
        train_loader:   instance of pytorch DataLoader
        device:         device to use
    """
    with torch.no_grad():
        flow.eval()
        print('Recalculating moving averages for batch norm layers.')
        flow.start_averaging()
        for h, x in train_loader:

            if device is not None:
                h = h.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)

            # Sample a noise realization
            # y_dist = torch.distributions.Normal(loc=h,
            #                                    scale=torch.ones_like(h))
            # y = y_dist.sample()

            y = h + torch.randn_like(h)

            # Flow
            _, _ = flow(x, y)
        flow.end_averaging()


def obtain_samples(flow, base_dist, y, nsamples, device=None):
    """Sample from the base distribution and apply MAF."""
    with torch.no_grad():
        flow.eval()
        u_samples = base_dist.sample([nsamples])
        if device is not None:
            y = torch.from_numpy(y).to(device).expand(nsamples, -1)
        x_samples, _ = flow.inverse(u_samples, y)
        return x_samples
