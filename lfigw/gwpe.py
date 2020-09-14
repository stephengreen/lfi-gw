import os

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

import argparse
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import corner
import csv
import time
import numpy as np
import h5py

from . import waveform_generator as wfg
from . import a_flows
from . import nde_flows
from . import cvae


class PosteriorModel(object):

    def __init__(self, model_dir=None, data_dir=None,
                 use_cuda=True):

        self.wfd = None
        self.model = None
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_type = None
        self.optimizer = None
        self.scheduler = None
        self.detectors = None
        self.train_history = []
        self.test_history = []
        self.train_kl_history = []
        self.test_kl_history = []

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def load_dataset(self, batch_size=512, detectors=None,
                     truncate_basis=None, snr_threshold=None,
                     distance_prior_fn=None, distance_prior=None,
                     bw_dstar=None):
        """Load database of waveforms and set up data loaders.

        Args:

            batch_size (int):  batch size for DataLoaders
        """

        if self.data_dir is None:
            raise NameError("Data directory must be specified."
                            " Store in attribute PosteriorModel.data_dir")

        # Load waveforms, already split into train and test sets
        self.wfd = wfg.WaveformDataset()
        self.wfd.load(self.data_dir)
        self.wfd.load_train(self.data_dir)

        # Set the detectors for training; useful if this is different from
        # stored detectors in WaveformDataset
        if self.detectors is not None:
            self.wfd.init_detectors(self.detectors)
        elif detectors is not None:
            # Only pay attention to argument if self.detectors not set
            self.wfd.init_detectors(detectors)
            self.detectors = detectors
        else:
            self.detectors = list(self.wfd.detectors.keys())

        if self.wfd.domain == 'RB':
            # Optionally, train the network with a truncation of the
            # reduced order basis.
            if truncate_basis is not None:
                self.wfd.truncate_basis(truncate_basis)

            # Additional initialization (time translations, whitening) for
            # reduced basis. This should be done *after* truncating the basis
            # to save time in generating time translation matrices.
            self.wfd.initialize_reduced_basis_aux()

            # Initialize the SNR threshold. This needs to be done after fully
            # initializing the reduced basis, so that the time translation
            # and whitening transformations are available.
            restandardize = False

            if snr_threshold is not None:
                print('Setting SNR threshold to {}.'.format(snr_threshold))
                self.wfd.snr_threshold = snr_threshold
                restandardize = True

            if distance_prior_fn is not None:
                print('Using distance prior function: {}'.format(
                      distance_prior_fn))
                self.wfd.distance_prior_fn = distance_prior_fn
                self.wfd.bw_dstar = bw_dstar
                restandardize = True

            if distance_prior is not None:
                print('Setting distance prior to {}'.format(distance_prior))
                self.wfd.prior['distance'] = distance_prior
                restandardize = True

            if restandardize:
                self.wfd.calculate_threshold_standardizations()

        # pytorch wrappers
        wfd_train = wfg.WaveformDatasetTorch(self.wfd, train=True)
        wfd_test = wfg.WaveformDatasetTorch(self.wfd, train=False)

        # DataLoader objects
        self.train_loader = DataLoader(
            wfd_train, batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=16,
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2**32-1)))
        self.test_loader = DataLoader(
            wfd_test, batch_size=batch_size, shuffle=False, pin_memory=True,
            num_workers=16,
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2**32-1)))

    def construct_model(self, model_type, existing=False, **kwargs):
        """Construct the neural network model.

        Args:

            model_type:     'maf' or 'cvae'
            wfd:            (Optional) If constructing the model from a
                            WaveformDataset, include this. Otherwise, all
                            arguments are passed through kwargs.

            kwargs:         Depends on the model_type

                'maf'   input_dim       Do not include with wfd
                        context_dim     Do not include with wfd
                        hidden_dims
                        nflows
                        batch_norm      (True)
                        bn_momentum     (0.9)
                        activation      ('elu')

                'cvae'  input_dim       Do not include with wfd
                        context_dim     Do not include with wfd
                        latent_dim      int
                        hidden_dims     same for encoder and decoder
                                        list of ints
                        encoder_full_cov (True)
                        decoder_full_cov (True)
                        activation      ('elu')
                        batch_norm      (False)

                        iaf             Either None, or a dictionary of
                                        hyperparameters describing the desired
                                        IAF. Keys should be:
                                            context_dim
                                            hidden_dims
                                            nflows

                        prior_maf     Either None, or a dictionary of
                                        hyperparameters describing the desired
                                        MAF. Keys should be:
                                            hidden_dims
                                            nflows
                                        Note that this is conditioned on
                                        the waveforms automatically.

            * it is recommended to only use one of iaf or prior_maf

                        decoder_maf     Either None, or a dictionary of
                                        hyperparameters describing the desired
                                        MAF. Keys should be:
                                            hidden_dims
                                            nflows
                                        Note that this is conditioned on
                                        the waveforms automatically.
        """

        if model_type == 'maf':
            model_creator = a_flows.MAFStack
        elif model_type == 'cvae':
            model_creator = cvae.CVAE
        elif model_type == 'nde':
            model_creator = nde_flows.create_NDE_model
        else:
            raise NameError('Invalid model type')

        if not existing:
            input_dim = self.wfd.nparams
            context_dim = self.wfd.context_dim
            self.model = model_creator(input_dim=input_dim,
                                       context_dim=context_dim,
                                       **kwargs)
        else:
            self.model = model_creator(**kwargs)

        # Base distribution for sampling
        if model_type == 'maf':
            base_dim = self.model.model_hyperparams['input_dim']
        elif model_type == 'cvae':
            base_dim = self.model.model_hyperparams['latent_dim']
        if model_type == 'maf' or model_type == 'cvae':
            self.base_dist = (torch.distributions.
                              MultivariateNormal(
                                  loc=torch.zeros(
                                      base_dim, device=self.device),
                                  covariance_matrix=torch.diag_embed(
                                      torch.ones(base_dim,
                                                 device=self.device))))

        # I would like to use the code below, but the KL divergence doesn't
        # work... Should be a workaround.

        # self.base_dist = torch.distributions.Independent(
        #     torch.distributions.Normal(
        #         loc=torch.zeros(base_dim, device=self.device),
        #         scale=torch.ones(base_dim, device=self.device)
        #         ),
        #     1
        # )

        self.model.to(self.device)

        self.model_type = model_type

    def initialize_training(self, lr=0.0001,
                            lr_annealing=True, anneal_method='step',
                            total_epochs=None,
                            steplr_step_size=80, steplr_gamma=0.5,
                            flow_lr=None):
        """Set up the optimizer and scheduler."""

        if self.model is None:
            raise NameError('Construct model before initializing training.')

        if (flow_lr is not None) and (self.model_type == 'cvae'):
            param_list = [
                {'params': self.model.encoder.parameters()},
                {'params': self.model.decoder.parameters()},
                {'params': self.model.prior_nn.parameters()}
            ]
            possible_flows = [self.model.iaf,
                              self.model.prior_maf,
                              self.model.decoder_maf]
            for flow in possible_flows:
                if flow is not None:
                    param_list.append({'params': flow.parameters(),
                                       'lr': flow_lr})
            self.optimizer = torch.optim.Adam(param_list, lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if lr_annealing is True:
            if anneal_method == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=steplr_step_size,
                    gamma=steplr_gamma)
            elif anneal_method == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_epochs,
                )
            elif anneal_method == 'cosineWR':
                self.scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.optimizer,
                        T_0=10,
                        T_mult=2
                    )
                )

        self.epoch = 1

    def save_model(self, filename='model.pt',
                   aux_filename='waveforms_supplementary.hdf5'):
        """Save a model and optimizer to file.

        Args:

            model:      model to be saved
            optimizer:  optimizer to be saved
            epoch:      current epoch number
            model_dir:  directory to save the model in
            filename:   filename for saved model
        """

        if self.model_dir is None:
            raise NameError("Model directory must be specified."
                            " Store in attribute PosteriorModel.model_dir")

        p = Path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)

        dict = {
            'model_type': self.model_type,
            'model_hyperparams': self.model.model_hyperparams,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'detectors': self.detectors
        }

        if self.scheduler is not None:
            dict['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(dict, p / filename)

        # Save any information about basis truncation or standardization in
        # another file.
        f = h5py.File(p / aux_filename, 'w')

        if self.wfd.domain == 'RB':
            f.attrs['Nrb'] = self.wfd.Nrb

            std_group = f.create_group('RB_std')
            for ifo, std, in self.wfd.basis.standardization_dict.items():
                std_group.create_dataset(ifo, data=std,
                                         compression='gzip',
                                         compression_opts=9)

        f.create_dataset('parameters_mean', data=self.wfd.parameters_mean)
        f.create_dataset('parameters_std', data=self.wfd.parameters_std)

        f.close()

    def load_model(self, filename='model.pt'):
        """Load a saved model.

        Args:

            filename:       File name
        """

        if self.model_dir is None:
            raise NameError("Model directory must be specified."
                            " Store in attribute PosteriorModel.model_dir")

        p = Path(self.model_dir)
        checkpoint = torch.load(p / filename, map_location=self.device)

        model_type = checkpoint['model_type']
        model_hyperparams = checkpoint['model_hyperparams']

        # Load model
        self.construct_model(model_type, existing=True, **model_hyperparams)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Load optimizer
        scheduler_present_in_checkpoint = ('scheduler_state_dict' in
                                           checkpoint.keys())

        # If the optimizer has more than 1 param_group, then we built it with
        # flow_lr different from lr
        if len(checkpoint['optimizer_state_dict']['param_groups']) > 1:
            flow_lr = (checkpoint['optimizer_state_dict']['param_groups'][-1]
                       ['initial_lr'])
        else:
            flow_lr = None
        self.initialize_training(lr_annealing=scheduler_present_in_checkpoint,
                                 flow_lr=flow_lr)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler_present_in_checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load history
        with open(p / 'history.txt', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.train_history.append(float(row[1]))
                self.test_history.append(float(row[2]))

        # Load KL history if cvae
        if self.model_type == 'cvae':
            with open(p / 'kl_history.txt', 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    self.train_kl_history.append(float(row[1]))
                    self.test_kl_history.append(float(row[2]))

        # Set the epoch to the correct value. This is needed to resume
        # training.
        self.epoch = checkpoint['epoch']

        # Store the list of detectors the model was trained with
        self.detectors = checkpoint['detectors']

        # Make sure the model is in evaluation mode
        self.model.eval()

    def train(self, epochs, output_freq=50, kl_annealing=True,
              snr_annealing=False):
        """Train the model.

        Args:
                epochs:     number of epochs to train for
                output_freq:    how many iterations between outputs
                kl_annealing:  for cvae, whether to anneal the kl loss
        """

        if self.wfd.extrinsic_at_train:
            add_noise = False
        else:
            add_noise = True

        for epoch in range(self.epoch, self.epoch + epochs):

            print('Learning rate: {}'.format(
                self.optimizer.state_dict()['param_groups'][0]['lr']))
            if self.model_type == 'maf':
                train_loss = a_flows.train_epoch(
                    self.model,
                    self.base_dist,
                    self.train_loader,
                    self.optimizer,
                    epoch,
                    self.device,
                    output_freq)
                a_flows.recalculate_moving_avgs(self.model,
                                                self.train_loader,
                                                self.device)
                test_loss = a_flows.test_epoch(
                    self.model,
                    self.base_dist,
                    self.test_loader,
                    self.device)

            elif self.model_type == 'nde':
                train_loss = nde_flows.train_epoch(
                    self.model,
                    self.train_loader,
                    self.optimizer,
                    epoch,
                    self.device,
                    output_freq,
                    add_noise,
                    snr_annealing)
                test_loss = nde_flows.test_epoch(
                    self.model,
                    self.test_loader,
                    epoch,
                    self.device,
                    add_noise,
                    snr_annealing)

            elif self.model_type == 'cvae':
                train_loss, train_reconstruction_loss, train_kl_loss = \
                    cvae.train_epoch(
                        self.model,
                        self.base_dist,
                        self.train_loader,
                        self.optimizer,
                        epoch,
                        self.device,
                        output_freq,
                        annealing=kl_annealing)
                cvae.recalculate_moving_avgs(self.model,
                                             self.train_loader,
                                             self.base_dist,
                                             self.device)
                test_loss, test_reconstruction_loss, test_kl_loss = \
                    cvae.test_epoch(
                        self.model,
                        self.base_dist,
                        self.test_loader,
                        epoch,
                        self.device,
                        annealing=kl_annealing)

                self.train_kl_history.append(train_kl_loss)
                self.test_kl_history.append(test_kl_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            self.epoch = epoch + 1
            self.train_history.append(train_loss)
            self.test_history.append(test_loss)

            # Log the history to file
            if self.model_dir is not None:
                p = Path(self.model_dir)
                p.mkdir(parents=True, exist_ok=True)

                # Make column headers if this is the first epoch
                if epoch == 1:
                    with open(p / 'history.txt', 'w') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow([epoch, train_loss, test_loss])
                    if self.model_type == 'cvae':
                        with open(p / 'kl_history.txt', 'w') as f:
                            writer = csv.writer(f, delimiter='\t')
                            writer.writerow(
                                [epoch, train_kl_loss, test_kl_loss])
                else:
                    with open(p / 'history.txt', 'a') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow([epoch, train_loss, test_loss])
                    if self.model_type == 'cvae':
                        with open(p / 'kl_history.txt', 'a') as f:
                            writer = csv.writer(f, delimiter='\t')
                            writer.writerow(
                                [epoch, train_kl_loss, test_kl_loss])

    def init_waveform_supp(self, aux_filename='waveforms_supplementary.hdf5'):

        p = Path(self.model_dir)

        try:
            f = h5py.File(p / aux_filename, 'r')
        except FileNotFoundError:
            return

        if self.wfd.domain == 'RB':

            # Truncate basis if necessary
            Nrb = f.attrs['Nrb']
            if Nrb != self.wfd.Nrb:
                self.wfd.basis.truncate(Nrb)
                self.wfd.Nrb = Nrb

            std_group = f['RB_std']
            for ifo in std_group.keys():
                self.wfd.basis.standardization_dict[ifo] = std_group[ifo][:]

        self.wfd.parameters_mean = f['parameters_mean'][:]
        self.wfd.parameters_std = f['parameters_std'][:]

        f.close()

    def evaluate(self, idx, nsamples=10000, plot=True):
        """Evaluate the model on a noisy waveform.

        Args:
            idx         index of the waveform, from a noisy waveform
                        database
            plot        whether to make a corner plot
        """

        if self.wfd is None:
            self.wfd = wfg.WaveformDataset()

        if self.wfd.noisy_test_waveforms is None:
            self.wfd.load_noisy_test_data(self.data_dir)
            if list(self.wfd.detectors.keys()) != self.detectors:
                raise Exception("Model trained on different number of "
                                "detectors than contained in test data.")
            self.init_waveform_supp()

        # y = self.wfd.noisy_test_waveforms[idx]
        params_true = self.wfd.noisy_waveforms_parameters[idx]

        if self.wfd.domain == 'RB':
            h_dict = {}
            for ifo, h_array in self.wfd.noisy_test_waveforms.items():
                h_FD = h_array[idx]
                h_RB = self.wfd.basis.fseries_to_basis_coefficients(h_FD)
                h_dict[ifo] = h_RB
            _, y = self.wfd.x_y_from_p_h(params_true, h_dict, add_noise=False)

        if self.model_type == 'maf':
            x_samples = a_flows.obtain_samples(
                self.model, self.base_dist, y, nsamples, self.device)
        elif self.model_type == 'nde':
            x_samples = nde_flows.obtain_samples(
                self.model, y, nsamples, self.device
            )
        elif self.model_type == 'cvae':
            x_samples = cvae.obtain_samples(
                self.model, self.base_dist, y, nsamples, self.device)

        x_samples = x_samples.cpu()

        params_samples = self.wfd.post_process_parameters(x_samples.numpy())
        # params_samples

        if plot:
            corner.corner(params_samples, truths=params_true,
                          labels=self.wfd.parameter_labels)
            plt.show()

        return params_samples


class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description=('Model the gravitational-wave parameter '
                     'posterior distribution with neural networks.'))

    # Since options are often combined, defined parent parsers here and pass
    # them as parents when defining ArgumentParsers.

    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('--data_dir', type=str, required=True)
    dir_parent_parser.add_argument('--model_dir', type=str, required=True)
    dir_parent_parser.add_argument('--no_cuda', action='store_false',
                                   dest='cuda')

    activation_parent_parser = argparse.ArgumentParser(add_help=None)
    activation_parent_parser.add_argument(
        '--activation', choices=['relu', 'leaky_relu', 'elu'], default='relu')

    train_parent_parser = argparse.ArgumentParser(add_help=None)
    train_parent_parser.add_argument(
        '--batch_size', type=int, default='512')
    train_parent_parser.add_argument('--lr', type=float, default='0.0001')
    train_parent_parser.add_argument('--lr_anneal_method',
                                     choices=['step', 'cosine', 'cosineWR'],
                                     default='step')
    train_parent_parser.add_argument('--no_lr_annealing', action='store_false',
                                     dest='lr_annealing')
    train_parent_parser.add_argument(
        '--steplr_gamma', type=float, default=0.5)
    train_parent_parser.add_argument('--steplr_step_size', type=int,
                                     default=80)
    train_parent_parser.add_argument('--flow_lr', type=float)
    train_parent_parser.add_argument('--epochs', type=int, required=True)
    train_parent_parser.add_argument(
        '--output_freq', type=int, default='50')
    train_parent_parser.add_argument('--no_save', action='store_false',
                                     dest='save')
    train_parent_parser.add_argument('--no_kl_annealing', action='store_false',
                                     dest='kl_annealing')
    train_parent_parser.add_argument('--detectors', nargs='+')
    train_parent_parser.add_argument('--truncate_basis', type=int)
    train_parent_parser.add_argument('--snr_threshold', type=float)
    train_parent_parser.add_argument('--distance_prior_fn',
                                     choices=['uniform_distance',
                                              'inverse_distance',
                                              'linear_distance',
                                              'inverse_square_distance',
                                              'bayeswave'])
    train_parent_parser.add_argument('--snr_annealing', action='store_true')
    train_parent_parser.add_argument('--distance_prior', type=float,
                                     nargs=2)
    train_parent_parser.add_argument('--bw_dstar', type=float)

    cvae_parent_parser = argparse.ArgumentParser(add_help=False)
    cvae_parent_parser.add_argument(
        '--latent_dim', type=int, required=True)
    cvae_parent_parser.add_argument('--hidden_dims', type=int,
                                    nargs='+', required=True)
    cvae_parent_parser.add_argument('--batch_norm', action='store_true')
    cvae_parent_parser.add_argument(
        '--prior_gaussian_nn', action='store_true')
    cvae_parent_parser.add_argument('--prior_full_cov', action='store_true')

    iaf_parent_parser = argparse.ArgumentParser(add_help=False)
    iaf_parent_parser.add_argument('--iaf.hidden_dims', type=int, nargs='+',
                                   required=True)
    context_group = iaf_parent_parser.add_mutually_exclusive_group(
        required=True)
    context_group.add_argument('--iaf.context_dim', type=int)
    context_group.add_argument('--iaf.context_y', action='store_true')
    iaf_parent_parser.add_argument('--iaf.nflows', type=int, required=True)
    iaf_parent_parser.add_argument('--iaf.batch_norm', action='store_true')
    iaf_parent_parser.add_argument('--iaf.bn_momentum', type=float,
                                   default=0.9)
    iaf_parent_parser.add_argument('--iaf.maf_parametrization',
                                   action='store_false',
                                   dest='iaf.iaf_parametrization')
    iaf_parent_parser.add_argument('--iaf.xcontext', action='store_true')
    iaf_parent_parser.add_argument('--iaf.ycontext', action='store_true')

    maf_prior_parent_parser = argparse.ArgumentParser(add_help=False)
    maf_prior_parent_parser.add_argument('--maf_prior.hidden_dims', type=int,
                                         nargs='+',
                                         required=True)
    maf_prior_parent_parser.add_argument('--maf_prior.nflows', type=int,
                                         required=True)
    maf_prior_parent_parser.add_argument('--maf_prior.no_batch_norm',
                                         action='store_false',
                                         dest='maf_prior.batch_norm')
    maf_prior_parent_parser.add_argument('--maf_prior.bn_momentum', type=float,
                                         default=0.9)
    maf_prior_parent_parser.add_argument('--maf_prior.iaf_parametrization',
                                         action='store_true')

    maf_decoder_parent_parser = argparse.ArgumentParser(add_help=False)
    maf_decoder_parent_parser.add_argument('--maf_decoder.hidden_dims',
                                           type=int,
                                           nargs='+',
                                           required=True)
    maf_decoder_parent_parser.add_argument('--maf_decoder.nflows',
                                           type=int,
                                           required=True)
    maf_decoder_parent_parser.add_argument('--maf_decoder.no_batch_norm',
                                           action='store_false',
                                           dest='maf_decoder.batch_norm')
    maf_decoder_parent_parser.add_argument('--maf_decoder.bn_momentum',
                                           type=float,
                                           default=0.9)
    maf_decoder_parent_parser.add_argument('--maf_decoder.iaf_parametrization',
                                           action='store_true')
    maf_decoder_parent_parser.add_argument('--maf_decoder.zcontext',
                                           action='store_true')

    # Subprograms

    mode_subparsers = parser.add_subparsers(title='mode', dest='mode')
    mode_subparsers.required = True

    train_parser = mode_subparsers.add_parser(
        'train', description=('Train a network.'))

    train_subparsers = train_parser.add_subparsers(dest='model_source')
    train_subparsers.required = True

    train_new_parser = train_subparsers.add_parser(
        'new', description=('Build and train a network.'))

    type_subparsers = train_new_parser.add_subparsers(dest='model_type')
    type_subparsers.required = True

    # Pure MAF

    maf_parser = type_subparsers.add_parser(
        'maf',
        description=('Build and train a MAF.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 train_parent_parser])
    maf_parser.add_argument('--hidden_dims', type=int, nargs='+',
                            required=True)
    maf_parser.add_argument('--nflows', type=int, required=True)
    maf_parser.add_argument(
        '--no_batch_norm', action='store_false', dest='batch_norm')
    maf_parser.add_argument('--bn_momentum', type=float, default=0.9)
    maf_parser.add_argument('--iaf_parametrization', action='store_true')

    # nde (curently just NSFC)

    nde_parser = type_subparsers.add_parser(
        'nde',
        description=('Build and train a flow from the nde package.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 train_parent_parser]
    )
    nde_parser.add_argument('--hidden_dims', type=int, required=True)
    nde_parser.add_argument('--nflows', type=int, required=True)
    nde_parser.add_argument('--batch_norm', action='store_true')
    nde_parser.add_argument('--nbins', type=int, required=True)
    nde_parser.add_argument('--tail_bound', type=float, default=1.0)
    nde_parser.add_argument('--apply_unconditional_transform',
                            action='store_true')
    nde_parser.add_argument('--dropout_probability', type=float, default=0.0)
    nde_parser.add_argument('--num_transform_blocks', type=int, default=2)
    nde_parser.add_argument('--base_transform_type', type=str,
                            choices=['rq-coupling', 'rq-autoregressive'],
                            default='rq-coupling')

    # Pure CVAE

    cvae_parser = type_subparsers.add_parser(
        'cvae',
        description=('Build and train a CVAE.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 train_parent_parser])
    cvae_parser.add_argument('--encoder_diag_cov', action='store_false',
                             dest='encoder_full_cov')
    cvae_parser.add_argument('--decoder_diag_cov', action='store_false',
                             dest='decoder_full_cov')

    # CVAE with IAF

    cvae_iaf_parser = type_subparsers.add_parser(
        'cvae+iaf',
        description=('Build and train a CVAE with IAF encoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 iaf_parent_parser,
                 train_parent_parser])
    cvae_iaf_parser.add_argument('--encoder_full_cov', action='store_true')
    cvae_iaf_parser.add_argument('--decoder_diag_cov', action='store_false',
                                 dest='decoder_full_cov')

    # CVAE with prior MAF

    cvae_maf_prior_parser = type_subparsers.add_parser(
        'cvae+maf_prior',
        description=('Build and train a CVAE with MAF prior.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 maf_prior_parent_parser,
                 train_parent_parser])
    cvae_maf_prior_parser.add_argument('--encoder_full_cov',
                                       action='store_true')
    cvae_maf_prior_parser.add_argument('--decoder_diag_cov',
                                       action='store_false',
                                       dest='decoder_full_cov')

    # CVAE with decoder MAF

    cvae_maf_decoder_parser = type_subparsers.add_parser(
        'cvae+maf_decoder',
        description=('Build and train a CVAE with MAF decoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 maf_decoder_parent_parser,
                 train_parent_parser])
    cvae_maf_decoder_parser.add_argument('--encoder_diag_cov',
                                         action='store_false',
                                         dest='encoder_full_cov')
    cvae_maf_decoder_parser.add_argument('--decoder_full_cov',
                                         action='store_true')

    # CVAE with IAF + MAF decoder

    cvae_iaf_maf_decoder_parser = type_subparsers.add_parser(
        'cvae+iaf+maf_decoder',
        description=('Build and train a CVAE with IAF encoder'
                     'and MAF decoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 iaf_parent_parser,
                 maf_decoder_parent_parser,
                 train_parent_parser])
    cvae_iaf_maf_decoder_parser.add_argument('--encoder_full_cov',
                                             action='store_true')
    cvae_iaf_maf_decoder_parser.add_argument('--decoder_full_cov',
                                             action='store_true')

    # CVAE with prior MAF + posterior MAF

    cvae_maf_prior_maf_decoder_parser = type_subparsers.add_parser(
        'cvae+maf_prior+maf_decoder',
        description=('Build and train a CVAE with MAF prior'
                     'and MAF decoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 maf_prior_parent_parser,
                 maf_decoder_parent_parser,
                 train_parent_parser])
    cvae_maf_prior_maf_decoder_parser.add_argument('--encoder_full_cov',
                                                   action='store_true')
    cvae_maf_prior_maf_decoder_parser.add_argument('--decoder_full_cov',
                                                   action='store_true')

    # CVAE with IAF + prior MAF + posterior MAF

    cvae_all_parser = type_subparsers.add_parser(
        'cvae+all',
        description=('Build and train a CVAE with IAF, MAF prior'
                     'and MAF decoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 iaf_parent_parser,
                 maf_prior_parent_parser,
                 maf_decoder_parent_parser,
                 train_parent_parser])
    cvae_all_parser.add_argument('--encoder_full_cov',
                                 action='store_true')
    cvae_all_parser.add_argument('--decoder_full_cov',
                                 action='store_true')

    train_subparsers.add_parser(
        'existing',
        description=('Load a network from file and continue training.'),
        parents=[dir_parent_parser, train_parent_parser])

    ns = Nestedspace()

    return parser.parse_args(namespace=ns)


def main():
    args = parse_args()

    if args.mode == 'train':

        print('Waveform directory', args.data_dir)
        print('Model directory', args.model_dir)
        pm = PosteriorModel(model_dir=args.model_dir,
                            data_dir=args.data_dir,
                            use_cuda=args.cuda)
        print('Device', pm.device)
        print('Loading dataset')
        pm.load_dataset(batch_size=args.batch_size,
                        detectors=args.detectors,
                        truncate_basis=args.truncate_basis,
                        snr_threshold=args.snr_threshold,
                        distance_prior_fn=args.distance_prior_fn,
                        distance_prior=args.distance_prior,
                        bw_dstar=args.bw_dstar)
        print('Detectors:', pm.detectors)

        if args.model_source == 'new':

            print('\nConstructing model of type', args.model_type)

            if args.model_type == 'maf':
                pm.construct_model(
                    'maf',
                    hidden_dims=args.hidden_dims,
                    nflows=args.nflows,
                    batch_norm=args.batch_norm,
                    bn_momentum=args.bn_momentum,
                    iaf_parametrization=args.iaf_parametrization,
                    activation=args.activation)

            elif args.model_type == 'nde':
                pm.construct_model(
                    'nde',
                    num_flow_steps=args.nflows,
                    base_transform_kwargs={
                        'hidden_dim': args.hidden_dims,
                        'num_transform_blocks': args.num_transform_blocks,
                        'activation': args.activation,
                        'dropout_probability': args.dropout_probability,
                        'batch_norm': args.batch_norm,
                        'num_bins': args.nbins,
                        'tail_bound': args.tail_bound,
                        'apply_unconditional_transform': args.apply_unconditional_transform,
                        'base_transform_type': args.base_transform_type
                    }
                )

            elif args.model_type == 'cvae':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+iaf':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    iaf={
                        'context_dim': args.iaf.context_dim,
                        'hidden_dims': args.iaf.hidden_dims,
                        'nflows': args.iaf.nflows,
                        'batch_norm': args.iaf.batch_norm,
                        'bn_momentum': args.iaf.bn_momentum,
                        'iaf_parametrization': args.iaf.iaf_parametrization
                    },
                    encoder_xcontext=args.iaf.xcontext,
                    encoder_ycontext=args.iaf.ycontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+maf_prior':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    prior_maf={
                        'hidden_dims': args.maf_prior.hidden_dims,
                        'nflows': args.maf_prior.nflows,
                        'batch_norm': args.maf_prior.batch_norm,
                        'bn_momentum': args.maf_prior.bn_momentum,
                        'iaf_parametrization':
                        args.maf_prior.iaf_parametrization
                    },
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+maf_decoder':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    decoder_maf={
                        'hidden_dims': args.maf_decoder.hidden_dims,
                        'nflows': args.maf_decoder.nflows,
                        'batch_norm': args.maf_decoder.batch_norm,
                        'bn_momentum': args.maf_decoder.bn_momentum,
                        'iaf_parametrization':
                        args.maf_decoder.iaf_parametrization
                    },
                    decoder_zcontext=args.maf_decoder.zcontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+iaf+maf_decoder':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    iaf={
                        'context_dim': args.iaf.context_dim,
                        'hidden_dims': args.iaf.hidden_dims,
                        'nflows': args.iaf.nflows,
                        'batch_norm': args.iaf.batch_norm,
                        'bn_momentum': args.iaf.bn_momentum,
                        'iaf_parametrization': args.iaf.iaf_parametrization
                    },
                    decoder_maf={
                        'hidden_dims': args.maf_decoder.hidden_dims,
                        'nflows': args.maf_decoder.nflows,
                        'batch_norm': args.maf_decoder.batch_norm,
                        'bn_momentum': args.maf_decoder.bn_momentum,
                        'iaf_parametrization':
                        args.maf_decoder.iaf_parametrization
                    },
                    encoder_xcontext=args.iaf.xcontext,
                    encoder_ycontext=args.iaf.ycontext,
                    decoder_zcontext=args.maf_decoder.zcontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+maf_prior+maf_decoder':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    prior_maf={
                        'hidden_dims': args.maf_prior.hidden_dims,
                        'nflows': args.maf_prior.nflows,
                        'batch_norm': args.maf_prior.batch_norm,
                        'bn_momentum': args.maf_prior.bn_momentum,
                        'iaf_parametrization':
                        args.maf_prior.iaf_parametrization
                    },
                    decoder_maf={
                        'hidden_dims': args.maf_decoder.hidden_dims,
                        'nflows': args.maf_decoder.nflows,
                        'batch_norm': args.maf_decoder.batch_norm,
                        'bn_momentum': args.maf_decoder.bn_momentum,
                        'iaf_parametrization':
                        args.maf_decoder.iaf_parametrization
                    },
                    decoder_zcontext=args.maf_decoder.zcontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+all':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    iaf={
                        'context_dim': args.iaf.context_dim,
                        'hidden_dims': args.iaf.hidden_dims,
                        'nflows': args.iaf.nflows,
                        'batch_norm': args.iaf.batch_norm,
                        'bn_momentum': args.iaf.bn_momentum,
                        'iaf_parametrization': args.iaf.iaf_parametrization
                    },
                    prior_maf={
                        'hidden_dims': args.maf_prior.hidden_dims,
                        'nflows': args.maf_prior.nflows,
                        'batch_norm': args.maf_prior.batch_norm,
                        'bn_momentum': args.maf_prior.bn_momentum,
                        'iaf_parametrization':
                        args.maf_prior.iaf_parametrization
                    },
                    decoder_maf={
                        'hidden_dims': args.maf_decoder.hidden_dims,
                        'nflows': args.maf_decoder.nflows,
                        'batch_norm': args.maf_decoder.batch_norm,
                        'bn_momentum': args.maf_decoder.bn_momentum,
                        'iaf_parametrization':
                        args.maf_decoder.iaf_parametrization
                    },
                    encoder_xcontext=args.iaf.xcontext,
                    encoder_ycontext=args.iaf.ycontext,
                    decoder_zcontext=args.maf_decoder.zcontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            print('\nInitial learning rate', args.lr)
            if args.lr_annealing is True:
                if args.lr_anneal_method == 'step':
                    print('Stepping learning rate by', args.steplr_gamma,
                          'every', args.steplr_step_size, 'epochs')
                elif args.lr_anneal_method == 'cosine':
                    print('Using cosine LR annealing.')
                elif args.lr_anneal_method == 'cosineWR':
                    print('Using cosine LR annealing with warm restarts.')
            else:
                print('Using constant learning rate. No annealing.')
            if args.flow_lr is not None:
                print('Autoregressive flows initial lr', args.flow_lr)
            pm.initialize_training(lr=args.lr,
                                   lr_annealing=args.lr_annealing,
                                   anneal_method=args.lr_anneal_method,
                                   total_epochs=args.epochs,
                                   # steplr=args.steplr,
                                   steplr_step_size=args.steplr_step_size,
                                   steplr_gamma=args.steplr_gamma,
                                   flow_lr=args.flow_lr)

        elif args.model_source == 'existing':

            print('Loading existing model')
            pm.load_model()

        print('\nModel hyperparameters:')
        for key, value in pm.model.model_hyperparams.items():
            if type(value) == dict:
                print(key)
                for k, v in value.items():
                    print('\t', k, '\t', v)
            else:
                print(key, '\t', value)

        if pm.model_type == 'cvae' and args.kl_annealing:
            print('\nUsing cyclic KL annealing')

        print('\nTraining for {} epochs'.format(args.epochs))

        print('Starting timer')
        start_time = time.time()

        pm.train(args.epochs,
                 output_freq=args.output_freq,
                 kl_annealing=args.kl_annealing,
                 snr_annealing=args.snr_annealing)

        print('Stopping timer.')
        stop_time = time.time()
        print('Training time (including validation): {} seconds'
              .format(stop_time - start_time))

        if args.save:
            print('Saving model')
            pm.save_model()

    print('Program complete')


if __name__ == "__main__":
    main()
