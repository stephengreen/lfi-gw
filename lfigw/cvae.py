import torch
import torch.nn as nn
import numpy
import sys

from .a_flows import MAFStack


numpy.set_printoptions(threshold=sys.maxsize)


def _strictly_tril_size(n):
    return n * (n-1) // 2


class Coder(nn.Module):

    def __init__(self, input_dim, context_dim, hidden_dims,
                 output_dim, output_context_dim=None,
                 activation=nn.ELU(), full_cov=True, batch_norm=False):
        super(Coder, self).__init__()

        # Hidden layers
        hidden_net_list = []
        hidden_net_list.append(
            nn.Linear(input_dim + context_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            hidden_net_list.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.hidden_net_list = nn.ModuleList(hidden_net_list)

        # Batch norm layers
        if batch_norm:
            self.bn_list = nn.ModuleList(
                [nn.BatchNorm1d(dim) for dim in hidden_dims]
            )

        # Output layers
        self.output_loc_net = nn.Linear(hidden_dims[-1], output_dim)
        self.output_log_diag_net = nn.Linear(hidden_dims[-1], output_dim)
        if full_cov:
            self.output_chol_net = nn.Linear(hidden_dims[-1],
                                             _strictly_tril_size(output_dim))
            lt_indices = torch.tril_indices(output_dim, output_dim, -1)
            self.register_buffer('lt_indices', lt_indices)

        # Additional output layer which will be taken as context for IAF layers
        self.output_context_net = None
        if output_context_dim is not None:
            self.output_context_net = nn.Linear(hidden_dims[-1],
                                                output_context_dim)

        eps = torch.tensor(0.00001)
        self.register_buffer('eps', eps)

        self.af = activation
        self.full_cov = full_cov
        self.batch_norm = batch_norm

    def forward(self, x, context=None):

        # Concatenate x with context
        if context is not None:
            hidden = torch.cat((x, context), dim=-1)
        else:
            hidden = x

        # Pass through hidden layers
        for i, hn in enumerate(self.hidden_net_list):
            hidden = self.af(hn(hidden))
            if self.batch_norm:
                bn = self.bn_list[i]
                hidden = bn(hidden)

        # Output layer defines a Gaussian
        loc = self.output_loc_net(hidden)
        diag = (torch.nn.functional.softplus(self.output_log_diag_net(hidden))
                + self.eps)
        if self.full_cov:
            diag = torch.diag_embed(diag)
            chol = torch.zeros_like(diag)
            chol[..., self.lt_indices[0], self.lt_indices[1]
                 ] = self.output_chol_net(hidden)
            chol = chol + diag
            dist = torch.distributions.MultivariateNormal(
                loc=loc, scale_tril=chol)
        else:
            dist = torch.distributions.Independent(
                torch.distributions.Normal(loc=loc, scale=diag),
                1
            )
            # dist = torch.distributions.MultivariateNormal(
            #     loc=loc, covariance_matrix=diag)

        # Additional context output for IAF
        if self.output_context_net is not None:
            output_context = self.output_context_net(hidden)
            return dist, output_context
        else:
            return dist


class CVAE(nn.Module):

    def __init__(self, input_dim, context_dim, latent_dim, hidden_dims,
                 encoder_full_cov=True, decoder_full_cov=True,
                 activation='elu',
                 iaf=None, decoder_maf=None, prior_maf=None,
                 decoder_zcontext=False,
                 encoder_xcontext=False, encoder_ycontext=False,
                 batch_norm=False,
                 prior_gaussian_nn=False, prior_full_cov=False):
        super(CVAE, self).__init__()

        if activation == 'elu':
            af = nn.ELU()
        elif activation == 'relu':
            af = nn.ReLU()
        elif activation == 'leaky_relu':
            af = nn.LeakyReLU()
        else:
            af = nn.ReLU()   # Default
            print('Invalid activation function specified. Using ReLU.')

        if iaf is not None:
            encoder_output_context_dim = iaf['context_dim']
            encoder_context_dim = encoder_output_context_dim
            if encoder_xcontext:
                encoder_context_dim += input_dim
            if encoder_ycontext:
                encoder_context_dim += context_dim
            self.iaf = MAFStack(latent_dim,
                                encoder_context_dim,
                                iaf['hidden_dims'],
                                iaf['nflows'],
                                batch_norm=iaf['batch_norm'],
                                bn_momentum=iaf['bn_momentum'],
                                activation=activation,
                                iaf_parametrization=iaf['iaf_parametrization'])
        else:
            encoder_output_context_dim = None
            self.iaf = None

        if prior_maf is not None:
            self.prior_maf = MAFStack(
                latent_dim,
                context_dim,
                prior_maf['hidden_dims'],
                prior_maf['nflows'],
                batch_norm=prior_maf['batch_norm'],
                bn_momentum=prior_maf['bn_momentum'],
                activation=activation,
                iaf_parametrization=prior_maf['iaf_parametrization'])
        else:
            self.prior_maf = None

        if decoder_maf is not None:
            if decoder_zcontext:
                decoder_context_dim = context_dim + latent_dim
            else:
                decoder_context_dim = context_dim
            self.decoder_maf = MAFStack(
                input_dim,
                decoder_context_dim,
                decoder_maf['hidden_dims'],
                decoder_maf['nflows'],
                batch_norm=decoder_maf['batch_norm'],
                bn_momentum=decoder_maf['bn_momentum'],
                activation=activation,
                iaf_parametrization=decoder_maf['iaf_parametrization'])
        else:
            self.decoder_maf = None

        self.encoder = Coder(input_dim=input_dim,
                             context_dim=context_dim,
                             hidden_dims=hidden_dims,
                             output_dim=latent_dim,
                             output_context_dim=encoder_output_context_dim,
                             activation=af,
                             full_cov=encoder_full_cov,
                             batch_norm=batch_norm)

        self.decoder = Coder(input_dim=latent_dim,
                             context_dim=context_dim,
                             hidden_dims=hidden_dims,
                             output_dim=input_dim,
                             activation=af,
                             full_cov=decoder_full_cov,
                             batch_norm=batch_norm)

        if prior_gaussian_nn:
            self.prior_nn = Coder(input_dim=context_dim,
                                  context_dim=0,
                                  hidden_dims=hidden_dims,
                                  output_dim=latent_dim,
                                  activation=af,
                                  full_cov=prior_full_cov,
                                  batch_norm=batch_norm)
        else:
            self.prior_nn = None

        self.model_hyperparams = {
            'input_dim': input_dim,
            'context_dim': context_dim,
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'encoder_full_cov': encoder_full_cov,
            'decoder_full_cov': decoder_full_cov,
            'activation': activation,
            'iaf': iaf,
            'prior_maf': prior_maf,
            'decoder_maf': decoder_maf,
            'decoder_zcontext': decoder_zcontext,
            'encoder_xcontext': encoder_xcontext,
            'encoder_ycontext': encoder_ycontext,
            'batch_norm': batch_norm,
            'prior_gaussian_nn': prior_gaussian_nn,
            'prior_full_cov': prior_full_cov
        }

    def forward(self, x, context, base_dist):
        """Forward pass for CVAE modeling p(x|context).

        Note that we are currently fixing the prior r1(z|y) to be
        independent of y.

        Args:
            x           tensor
            context     tensor
            base_dist     prior distribution over latent variables, r1(z)
                            ** unused if use prior_nn
        """

        # Posterior over latent space, q(z|x,y) and possible context for IAF
        if self.iaf is not None:
            q_dist, iaf_context = self.encoder(x, context)
        else:
            q_dist = self.encoder(x, context)

        # Sample from q
        z = q_dist.rsample()

        # Flow through any IAF layers to arrive at final latent space
        if self.iaf is not None:
            if self.model_hyperparams['encoder_xcontext']:
                iaf_context = torch.cat((x, iaf_context), dim=-1)
            if self.model_hyperparams['encoder_ycontext']:
                iaf_context = torch.cat((context, iaf_context), dim=-1)
            zT, iaf_log_det = self.iaf(z, iaf_context)
        else:
            zT = z
            iaf_log_det = torch.zeros_like(z[..., 0])

        # Decode, r2(x|z,y)
        r2_dist = self.decoder(zT, context)

        # Flow the prior
        if self.prior_maf is not None:
            epsilon, prior_maf_log_det = self.prior_maf(zT, context)
        else:
            epsilon = zT
            prior_maf_log_det = torch.zeros_like(zT[..., 0])

        # Return the reconstruction loss and the KL loss
        #
        # reconstruction loss = - E_{q(z|x,y} log r2(x|z,y)
        #       where we take 1 MC sample to evaluate this
        #
        # KL loss = D_{KL}( q(z|x,y) || r(z|y) )
        #         = E_{q(z|x,y)} ( log q(z|x,y) - log r1(z) )

        #
        # Reconstruction loss
        #
        # Flow the final output back through any decoder MAF to the space over
        # which r2 is defined.
        #

        if self.decoder_maf is not None:
            if self.model_hyperparams['decoder_zcontext']:
                decoder_maf_context = torch.cat((context, zT), dim=-1)
            else:
                decoder_maf_context = context
            x0, decoder_maf_log_det = self.decoder_maf(x, decoder_maf_context)
            reconstruction_loss = - (r2_dist.log_prob(x0) -
                                     decoder_maf_log_det)
        else:
            reconstruction_loss = - r2_dist.log_prob(x)

        #
        # KL loss
        #

        # If we are using a neural network for the gaussian prior, compute the
        # distribution r1(z|y)
        if self.prior_nn is not None:
            prior_gaussian_dist = self.prior_nn(context)
        else:
            prior_gaussian_dist = base_dist

        if (self.iaf is not None) or (self.prior_maf is not None):
            # Single sample Monte Carlo estimate
            kl_loss = ((q_dist.log_prob(z) + iaf_log_det) -
                       (prior_gaussian_dist.log_prob(epsilon) -
                        prior_maf_log_det))
        else:
            # KL divergence known analytically between Gaussians
            kl_loss = torch.distributions.kl_divergence(
                q_dist, prior_gaussian_dist)

        return reconstruction_loss, kl_loss


anneal_start = 1
anneal_duration = 3
kl_weight_start = 0.00001
anneal_cycles = 4


def kl_weight_schedule(epoch, quiet=False):
    if epoch < 2 * anneal_duration * anneal_cycles:
        epoch = (epoch - 1) % (2 * anneal_duration) + 1
    if epoch < anneal_start:
        weight = kl_weight_start
    elif epoch < anneal_start + anneal_duration:
        weight = kl_weight_start + \
            (1.0 - kl_weight_start) * (epoch - anneal_start) / anneal_duration
    else:
        weight = 1.0
    if not quiet:
        print('Setting KL weight to {}'.format(weight))
    return weight


def train_epoch(model, base_dist, train_loader, optimizer, epoch,
                device=None, output_freq=50, annealing=True):
    """Train CVAE for 1 epoch

    Args:
        model:          instance of CVAE
        base_dist:        r1(z) prior distribution
        train_loader:   instance of pytorch DataLoader
        optimizer:      optimizer instance
        epoch:          int, epoch number
        device:         device to use
        output_freq:    number of batches between printed outputs
                            None suppresses outputs
        annealing:      whether to anneal the KL loss

    Returns:
        average reconstruction loss and kl loss over epoch
    """

    model.train()
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0

    # KL weight annealing. This is needed to avoid posterior collapse.
    if annealing:
        kl_weight = torch.tensor(kl_weight_schedule(epoch)).to(device)
    else:
        kl_weight = torch.tensor(1.0).to(device)

    for batch_idx, (h, x) in enumerate(train_loader):
        optimizer.zero_grad()

        if device is not None:
            h = h.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)

        # Sample a noise realization
        y = h + torch.randn_like(h)

        reconstruction_loss, kl_loss = model(x, y, base_dist)

        # Keep track of total of each loss
        total_reconstruction_loss += reconstruction_loss.sum()
        total_kl_loss += kl_loss.sum()

        reconstruction_loss = reconstruction_loss.mean()
        kl_loss = kl_loss.mean()

        # Total loss weights KL term by kl_weight, and also ignores it if less
        # than 0.2.
        loss = (reconstruction_loss + kl_weight *
                torch.max(kl_loss.new_tensor(0.2), kl_loss))

        # Backpropagate and take a step
        loss.backward()
        optimizer.step()

        if (output_freq is not None) and (batch_idx % output_freq == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t=\t'
                  'Reconstruction loss: {:.4f}\t +'
                  '\t (KL weight) * KL loss: {:.4f}'.format(
                      epoch, batch_idx *
                      train_loader.batch_size, len(train_loader.dataset),
                      100. * batch_idx / len(train_loader),
                      loss.item(), reconstruction_loss.item(), kl_loss.item()))

    avg_reconstruction_loss = (total_reconstruction_loss.item() /
                               len(train_loader.dataset))
    avg_kl_loss = (total_kl_loss.item() /
                   len(train_loader.dataset))
    avg_loss = avg_reconstruction_loss + kl_weight.item() * avg_kl_loss

    print('Train Epoch: {} \tAverage Loss: {:.4f}\t=\t'
          'Reconstruction loss: {:.4f}\t +'
          '\t (KL weight) * KL loss: {:.4f}'.format(
              epoch, avg_loss,
              avg_reconstruction_loss, avg_kl_loss))

    return avg_loss, avg_reconstruction_loss, avg_kl_loss


def test_epoch(model, base_dist, test_loader, epoch,
               device=None, annealing=False):
    """Calculate validation loss.

    Args:
        model:          instance of CVAE
        base_dist:        r1(z) prior distribution
        test_loader:    instance of pytorch DataLoader
        device:         device to use
        annealing:      whether to anneal the KL loss

    Returns:
        average reconstruction loss and kl loss over test_loader
    """

    # KL weight annealing. This is needed to avoid posterior collapse.
    if annealing:
        kl_weight = torch.tensor(
            kl_weight_schedule(epoch, quiet=True)).to(device)
    else:
        kl_weight = torch.tensor(1.0).to(device)

    with torch.no_grad():

        model.eval()
        total_reconstruction_loss = 0.0
        total_kl_loss = 0.0

        for h, x in test_loader:

            if device is not None:
                h = h.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)

            # Sample a noise realization
            y = h + torch.randn_like(h)

            reconstruction_loss, kl_loss = model(x, y, base_dist)

            # Keep track of total of each loss
            total_reconstruction_loss += reconstruction_loss.sum()
            total_kl_loss += kl_loss.sum()

        avg_reconstruction_loss = (total_reconstruction_loss.item() /
                                   len(test_loader.dataset))
        avg_kl_loss = (total_kl_loss.item() /
                       len(test_loader.dataset))
        avg_loss = avg_reconstruction_loss + kl_weight.item() * avg_kl_loss

        print('Test set: Average Loss: {:.4f}\t=\t'
              'Reconstruction loss: {:.4f}\t +'
              '\t (KL weight) * KL loss: {:.4f}\n'.format(
                  avg_loss,
                  avg_reconstruction_loss, avg_kl_loss))

        return avg_loss, avg_reconstruction_loss, avg_kl_loss


def recalculate_moving_avgs(model, train_loader, base_dist, device):
    """Reset the moving averages of the batch norm layers contained
    within any MAF flows to be the averages over the entire training set.

    This is necessary in order to achieve good performance on test set.

    Args:
        model:          instance of CVAE
        train_loader:   instance of pytorch DataLoader
        base_dist:      base distribution
        device:         device to use
    """
    with torch.no_grad():

        flow_list = []
        for flow in [model.iaf, model.prior_maf, model.decoder_maf]:
            if (flow is not None) and (flow.batch_norm):
                flow_list.append(flow)

        if len(flow_list) != 0:

            print('Recalculating moving averages for batch norm layers.')

            model.eval()
            for flow in flow_list:
                flow.start_averaging()

            for h, x in train_loader:

                if device is not None:
                    h = h.to(device, non_blocking=True)
                    x = x.to(device, non_blocking=True)

                y = h + torch.randn_like(h)

                # Pass through CVAE
                _, _ = model(x, y, base_dist)

            for flow in flow_list:
                flow.end_averaging()


def obtain_samples(model, base_dist, y, nsamples, device=None):
    """Sample from prior r1(z|y), then from r2(x|z,y).

    r1(z|y) reduces to base_dist if the model has no prior_nn."""

    with torch.no_grad():
        model.eval()

        if device is not None:
            y = torch.from_numpy(y).to(device).expand(nsamples, -1)
        else:
            y = torch.from_numpy(y).expand(nsamples, -1)

        # Sample from r1(z|y), or from untrainable base_dist, depending on
        # network
        if model.prior_nn is not None:
            z_samples = model.prior_nn(y).sample()
        else:
            z_samples = base_dist.sample([nsamples])

        # If there are MAF layers after the prior, flow through them
        if model.prior_maf is not None:
            z_samples, _ = model.prior_maf.inverse(z_samples, y)

        r2_dist = model.decoder(z_samples, y)
        x_samples = r2_dist.sample()

        # If there are MAF layers after decoder, flow through them
        if model.decoder_maf is not None:
            if model.model_hyperparams['decoder_zcontext']:
                decoder_maf_context = torch.cat((y, z_samples), dim=-1)
            else:
                decoder_maf_context = y
            x_samples, _ = model.decoder_maf.inverse(x_samples,
                                                     decoder_maf_context)

        return x_samples
