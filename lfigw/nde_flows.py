from nflows import distributions, flows, transforms, utils
import torch
from torch.nn import functional as F
import nflows.nn.nets as nn_


def create_linear_transform(param_dim):
    """Create the composite linear transform PLU.

    Arguments:
        input_dim {int} -- dimension of the space

    Returns:
        Transform -- nde.Transform object
    """

    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=param_dim),
        transforms.LULinear(param_dim, identity_init=True)
    ])


def create_base_transform(i,
                          param_dim,
                          context_dim=None,
                          hidden_dim=512,
                          num_transform_blocks=2,
                          activation='relu',
                          dropout_probability=0.0,
                          batch_norm=False,
                          num_bins=8,
                          tail_bound=1.,
                          apply_unconditional_transform=False,
                          base_transform_type='rq-coupling'
                          ):
    """Build a base NSF transform of x, conditioned on y.

    This uses the PiecewiseRationalQuadraticCoupling transform or
    the MaskedPiecewiseRationalQuadraticAutoregressiveTransform, as described
    in the Neural Spline Flow paper (https://arxiv.org/abs/1906.04032).

    Code is adapted from the uci.py example from
    https://github.com/bayesiains/nsf.

    A coupling flow fixes half the components of x, and applies a transform
    to the remaining components, conditioned on the fixed components. This is
    a restricted form of an autoregressive transform, with a single split into
    fixed/transformed components.

    The transform here is a neural spline flow, where the flow is parametrized
    by a residual neural network that depends on x_fixed and y. The residual
    network consists of a sequence of two-layer fully-connected blocks.

    Arguments:
        i {int} -- index of transform in sequence
        param_dim {int} -- dimensionality of x

    Keyword Arguments:
        context_dim {int} -- dimensionality of y (default: {None})
        hidden_dim {int} -- number of hidden units per layer (default: {512})
        num_transform_blocks {int} -- number of transform blocks comprising the
                                      transform (default: {2})
        activation {str} -- activation function (default: {'relu'})
        dropout_probability {float} -- probability of dropping out a unit
                                       (default: {0.0})
        batch_norm {bool} -- whether to use batch normalization
                             (default: {False})
        num_bins {int} -- number of bins for the spline (default: {8})
        tail_bound {[type]} -- [description] (default: {1.})
        apply_unconditional_transform {bool} -- whether to apply an
                                                unconditional transform to
                                                fixed components
                                                (default: {False})

        base_transform_type {str} -- type of base transform
                                     ([rq-coupling], rq-autoregressive)

    Returns:
        Transform -- the NSF transform
    """

    if activation == 'elu':
        activation_fn = F.elu
    elif activation == 'relu':
        activation_fn = F.relu
    elif activation == 'leaky_relu':
        activation_fn = F.leaky_relu
    else:
        activation_fn = F.relu   # Default
        print('Invalid activation function specified. Using ReLU.')

    if base_transform_type == 'rq-coupling':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)),
            transform_net_create_fn=(lambda in_features, out_features:
                                    nn_.ResidualNet(
                                        in_features=in_features,
                                        out_features=out_features,
                                        hidden_features=hidden_dim,
                                        context_features=context_dim,
                                        num_blocks=num_transform_blocks,
                                        activation=activation_fn,
                                        dropout_probability=dropout_probability,
                                        use_batch_norm=batch_norm
                                    )
                                    ),
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform
        )

    elif base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=param_dim,
            hidden_features=hidden_dim,
            context_features=context_dim,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=activation_fn,
            dropout_probability=dropout_probability,
            use_batch_norm=batch_norm
        )

    else:
        raise ValueError


def create_transform(num_flow_steps,
                     param_dim,
                     context_dim,
                     base_transform_kwargs):
    """Build a sequence of NSF transforms, which maps parameters x into the
    base distribution u (noise). Transforms are conditioned on strain data y.

    Note that the forward map is f^{-1}(x, y).

    Each step in the sequence consists of
        * A linear transform of x, which in particular permutes components
        * A NSF transform of x, conditioned on y.
    There is one final linear transform at the end.

    This function was adapted from the uci.py example in
    https://github.com/bayesiains/nsf

    Arguments:
        num_flow_steps {int} -- number of transforms in sequence
        param_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        base_transform_kwargs {dict} -- hyperparameters for NSF step

    Returns:
        Transform -- the constructed transform
    """

    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(param_dim),
            create_base_transform(i, param_dim, context_dim=context_dim,
                                  **base_transform_kwargs)
        ]) for i in range(num_flow_steps)
    ] + [
        create_linear_transform(param_dim)
    ])
    return transform


def create_NDE_model(input_dim, context_dim, num_flow_steps,
                     base_transform_kwargs):
    """Build NSF (neural spline flow) model. This uses the nsf module
    available at https://github.com/bayesiains/nsf.

    This models the posterior distribution p(x|y).

    The model consists of
        * a base distribution (StandardNormal, dim(x))
        * a sequence of transforms, each conditioned on y

    Arguments:
        input_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        num_flow_steps {int} -- number of sequential transforms
        base_transform_kwargs {dict} -- hyperparameters for transform steps

    Returns:
        Flow -- the model
    """

    distribution = distributions.StandardNormal((input_dim,))
    transform = create_transform(
        num_flow_steps, input_dim, context_dim, base_transform_kwargs)
    flow = flows.Flow(transform, distribution)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        'input_dim': input_dim,
        'num_flow_steps': num_flow_steps,
        'context_dim': context_dim,
        'base_transform_kwargs': base_transform_kwargs
    }

    return flow


anneal_duration = 50
anneal_max = 3.0


def anneal_schedule(epoch, quiet=False):
    if epoch <= anneal_duration:
        exponent = anneal_max * (anneal_duration - epoch + 1) / anneal_duration
    else:
        exponent = 0.0
    if not quiet:
        print('Setting annealing exponent to {}.'.format(exponent))
    return exponent


def train_epoch(flow, train_loader, optimizer, epoch,
                device=None,
                output_freq=50, add_noise=True, annealing=False):
    """Train model for one epoch.

    Arguments:
        flow {Flow} -- NSF model
        train_loader {DataLoader} -- train set data loader
        optimizer {Optimizer} -- model optimizer
        epoch {int} -- epoch number

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        output_freq {int} -- frequency for printing status (default: {50})

    Returns:
        float -- average train loss over epoch
    """

    flow.train()
    train_loss = 0.0
    total_weight = 0.0

    if annealing:
        anneal_exponent = anneal_schedule(epoch)
    else:
        anneal_exponent = 0.0

    # Change the sampling properties of the dataset over time

    if annealing:
        wfd = train_loader.dataset.wfd

        snr_threshold = 2 * anneal_exponent
        if snr_threshold > 0.0:
            print('SNR threshold: {}'.format(snr_threshold))
            wfd.snr_threshold = snr_threshold
        else:
            wfd.snr_threshold = None

        if anneal_exponent >= 2.0:
            wfd.importance_sampling = 'inverse_distance'
        elif anneal_exponent >= 1.0:
            wfd.importance_sampling = 'uniform_distance'
        else:
            wfd.importance_sampling = 'linear_distance'
        print('Importance sampling: {}'.format(wfd.importance_sampling))

        snr_threshold = torch.tensor(snr_threshold).to(device)

    anneal_exponent = torch.tensor(anneal_exponent).to(device)

    for batch_idx, (h, x, w, snr) in enumerate(train_loader):
        optimizer.zero_grad()

        if device is not None:
            h = h.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            snr = snr.to(device, non_blocking=True)

        if add_noise:
            # Sample a noise realization
            y = h + torch.randn_like(h)
            print('Should not be here')
        else:
            y = h

        # Compute log prob
        loss = - flow.log_prob(x, context=y)

        if anneal_exponent > 0.0:
            anneal_factor = (snr - snr_threshold) ** anneal_exponent
        else:
            anneal_factor = torch.tensor(1.0).to(device)

        loss = loss * anneal_factor

        # Keep track of total loss. w is a weight to be applied to each
        # element.
        train_loss += (w * loss).sum()
        total_weight += w.sum()

        # loss = (w * loss).sum() / w.sum()
        loss = (w * loss).mean()

        loss.backward()
        optimizer.step()

        if (output_freq is not None) and (batch_idx % output_freq == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx *
                train_loader.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    train_loss = train_loss.item() / len(train_loader.dataset)
    # train_loss = train_loss.item() / total_weight.item()
    print('Train Epoch: {} \tAverage Loss: {:.4f}'.format(
        epoch, train_loss))

    return train_loss


def test_epoch(flow, test_loader, epoch, device=None, add_noise=True,
               annealing=False):
    """Calculate test loss for one epoch.

    Arguments:
        flow {Flow} -- NSF model
        test_loader {DataLoader} -- test set data loader

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPu) (default: {None})

    Returns:
        float -- test loss
    """

    if annealing:
        anneal_exponent = anneal_schedule(epoch, quiet=True)
    else:
        anneal_exponent = 0.0

    snr_threshold = 2 * anneal_exponent

    anneal_exponent = torch.tensor(anneal_exponent).to(device)
    snr_threshold = torch.tensor(snr_threshold).to(device)

    with torch.no_grad():
        flow.eval()
        test_loss = 0.0
        total_weight = 0.0
        for h, x, w, snr in test_loader:

            if device is not None:
                h = h.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)
                snr = snr.to(device, non_blocking=True)

            if add_noise:
                # Sample a noise realization
                y = h + torch.randn_like(h)
            else:
                y = h

            # Compute log prob
            loss = - flow.log_prob(x, context=y)

            if anneal_exponent > 0.0:
                anneal_factor = (snr - snr_threshold) ** anneal_exponent
            else:
                anneal_factor = torch.tensor(1.0).to(device)

            loss = loss * anneal_factor

            # Keep track of total loss
            test_loss += (w * loss).sum()
            total_weight += w.sum()

        test_loss = test_loss.item() / len(test_loader.dataset)
        # test_loss = test_loss.item() / total_weight.item()
        print('Test set: Average loss: {:.4f}\n'
              .format(test_loss))

        return test_loss


def obtain_samples(flow, y, nsamples, device=None, batch_size=512):
    """Draw samples from the posterior.

    Arguments:
        flow {Flow} -- NSF model
        y {array} -- strain data
        nsamples {int} -- number of samples desired

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        batch_size {int} -- batch size for sampling (default: {512})

    Returns:
        Tensor -- samples
    """

    with torch.no_grad():
        flow.eval()

        y = torch.from_numpy(y).unsqueeze(0).to(device)

        num_batches = nsamples // batch_size
        num_leftover = nsamples % batch_size

        samples = [flow.sample(batch_size, y) for _ in range(num_batches)]
        if num_leftover > 0:
            samples.append(flow.sample(num_leftover, y))

        # The batching in the nsf package seems screwed up, so we had to do it
        # ourselves, as above. They are concatenating on the wrong axis.

        # samples = flow.sample(nsamples, context=y, batch_size=batch_size)

        return torch.cat(samples, dim=1)[0]
