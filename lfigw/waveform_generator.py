from .reduced_basis import SVDBasis
import h5py
import numpy as np
from pathlib import Path
import json
import functools
from tqdm import tqdm

from pycbc.waveform import (get_td_waveform, get_fd_waveform,
                            get_waveform_filter_length_in_time)
from pycbc.types.frequencyseries import FrequencySeries
import pycbc.psd
# from pycbc.waveform.utils import fd_taper

from pycbc.detector import Detector

from lalsimulation import (SimInspiralTransformPrecessingNewInitialConditions,
                           SimInspiralChooseFDWaveform,
                           SimInspiralFD,
                           SimInspiralImplementedFDApproximants,
                           GetApproximantFromString)
from lal import MSUN_SI, REARTH_SI, C_SI, PC_SI

import torch
from torch.utils.data import Dataset

from .bayeswave_prior import inverse_cdf as bw_inverse_cdf
from .bayeswave_prior import pdf as bw_pdf

# There seems to be an issue accessing IERS website. Temporary fix.
from astropy.utils import iers
iers.conf.auto_download = False

TIME_TRANSLATION_PTS = 1001


def source_frame_to_radiation(theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                              a_1, a_2, mass_1, mass_2, f_ref, phase):

    mass_1_SI = mass_1 * MSUN_SI
    mass_2_SI = mass_2 * MSUN_SI

    # Following bilby code

    if ((a_1 == 0.0 or tilt_1 in [0, np.pi])
            and (a_2 == 0.0 or tilt_2 in [0, np.pi])):
        spin_1x = 0.0
        spin_1y = 0.0
        spin_1z = a_1 * np.cos(tilt_1)
        spin_2x = 0.0
        spin_2y = 0.0
        spin_2z = a_2 * np.cos(tilt_2)
        iota = theta_jn
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
            SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                a_1, a_2, mass_1_SI, mass_2_SI, f_ref, phase
            )
        )
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z


def is_fd_waveform(approximant):
    """Return whether the approximant is implemented in FD.

    Args:
        approximant (str): name of approximant
    """
    # LAL refers to approximants by an index
    lal_num = GetApproximantFromString(approximant)
    return bool(SimInspiralImplementedFDApproximants(lal_num))


def m1_m2_from_M_q(M, q):
    """Compute individual masses from total mass and mass ratio.

    Choose m1 >= m2.

    Arguments:
        M {float} -- total mass
        q {mass ratio} -- mass ratio, 0.0< q <= 1.0

    Returns:
        (float, float) -- (mass_1, mass_2)
    """

    m1 = M / (1.0 + q)
    m2 = q * m1

    return m1, m2


def M_q_from_m1_m2(m1, m2):

    M = m1 + m2
    q = m2 / m1

    return M, q


class WaveformDataset(object):
    """Contains a database of waveforms from which to train a model.
    """

    def __init__(self, spins=True, inclination=True, spins_aligned=True,
                 detectors=['H1', 'L1', 'V1'], domain='TD',
                 extrinsic_at_train=False):

        # Set up indices for parameters
        param_idx = dict(mass_1=0, mass_2=1, phase=2, time=3, distance=4)
        nparams = 5
        if spins:
            if spins_aligned:
                param_idx['chi_1'] = nparams
                param_idx['chi_2'] = nparams + 1
                nparams += 2
            else:
                if inclination is not True:
                    raise Exception("Precession requires nonzero inclination.")
                param_idx['a_1'] = nparams
                param_idx['a_2'] = nparams + 1
                param_idx['tilt_1'] = nparams + 2
                param_idx['tilt_2'] = nparams + 3
                param_idx['phi_12'] = nparams + 4
                param_idx['phi_jl'] = nparams + 5
                nparams += 6
        if inclination:
            param_idx['theta_jn'] = nparams
            param_idx['psi'] = nparams + 1
            nparams += 2
        param_idx['ra'] = nparams
        param_idx['dec'] = nparams + 1
        nparams += 2
        self.param_idx = param_idx
        self.nparams = nparams

        # Default prior ranges
        self.prior = dict(mass_1=[10.0, 80.0],  # solar masses
                          mass_2=[10.0, 80.0],
                          # M=[25.0, 100.0],
                          # q=[0.125, 1.0],
                          phase=[0.0, 2*np.pi],
                          time=[-0.1, 0.1],  # seconds
                          distance=[100.0, 4000.0],  # Mpc
                          chi_1=[-1.0, 1.0],
                          chi_2=[-1.0, 1.0],
                          a_1=[0.0, 0.99],
                          a_2=[0.0, 0.99],
                          tilt_1=[0.0, np.pi],
                          tilt_2=[0.0, np.pi],
                          phi_12=[0.0, 2*np.pi],
                          phi_jl=[0.0, 2*np.pi],
                          theta_jn=[0.0, np.pi],
                          psi=[0.0, np.pi],
                          ra=[0.0, 2*np.pi],
                          dec=[-np.pi/2.0, np.pi/2.0])
        # self.set_m1_m2_ranges_from_M_q_ranges()

        # Whether to apply extrinsic parameters at train time or at dataset
        # preparation time.
        self.extrinsic_at_train = extrinsic_at_train
        self.extrinsic_params = ['time', 'distance',
                                 'psi', 'ra', 'dec']

        # Fiducial values for extrinsic parameters
        #
        # Note that the extrinsic parameters psi, ra, dec are simply ignored if
        # extrinsic_at_train is true. They don't need fiducial values.
        self.fiducial_params = dict(time=0.0,
                                    distance=1000.0)

        self.parameters_latex_dict = dict(mass_1=r'$m_1$',
                                          mass_2=r'$m_2$',
                                          phase=r'$\phi_c$',
                                          time=r'$t_c$',
                                          distance=r'$d_L$',
                                          chi_1=r'$\chi_1$',
                                          chi_2=r'$\chi_2$',
                                          a_1=r'$a_1$',
                                          a_2=r'$a_2$',
                                          tilt_1=r'$t_1$',
                                          tilt_2=r'$t_2$',
                                          phi_12=r'$\phi_{12}$',
                                          phi_jl=r'$\phi_{jl}$',
                                          theta_jn=r'$\theta_{JN}$',
                                          psi=r'$\psi$',
                                          ra=r'$\alpha$',
                                          dec=r'$\delta$')

        self.spins = spins
        self.spins_aligned = spins_aligned
        self.inclination = inclination
        self.init_detectors(detectors)

        # frequency at which source frame spin parameters are defined
        self.f_ref = 20.0

        # Default waveform parameters
        self.f_min = 8.0  # Hertz
        self.sampling_rate = 2048.0
        self.time_duration = 4.0  # seconds
        self.approximant = 'IMRPhenomPv2'
        self.ref_time = 100000000

        self.f_min_psd = 20.0  # Make sure this is smaller than f_min?
        self.psd_names = dict(H1='aLIGODesignSensitivityP1200087',
                              L1='aLIGODesignSensitivityP1200087',
                              V1='AdVDesignSensitivityP1200087',
                              ref='aLIGODesignSensitivityP1200087')
        self.psd = dict(H1={},
                        L1={},
                        V1={},
                        ref={})

        # Note that much of the code depends on detectory ordering in the
        # dictionary. Do not change the ordering. As of Python 3.7 dictionaries
        # are ordered.

        if domain in ('FD', 'RB'):
            self.f_min = 20.0
        self.domain = domain

        # Number of reduced basis elements
        if domain == 'RB':
            self.Nrb = 200

        # Initialize arrays
        self.parameters = None
        self.hp = None
        self.hc = None
        self.h_detector = {}
        self.train_selection = None
        self.test_selection = None
        self.noisy_test_waveforms = None
        self.noisy_waveforms_parameters = None

        # SNR threshold / changing distance prior
        self.snr_threshold = None
        self.distance_buffer = 50.0
        self.distance_prior_fn = None
        self.distance_power = 2.0
        self.bw_dstar = 500.0

        # These get set for real events
        self.window_factor = 1.0
        self.event = None
        self.event_dir = None

    @property
    def f_max(self):
        """Set the maximum frequency to half the sampling rate."""
        return self.sampling_rate / 2.0

    @f_max.setter
    def f_max(self, f_max):
        self.sampling_rate = 2.0 * f_max

    @property
    def delta_t(self):
        return 1.0 / self.sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t):
        self.sampling_rate = 1.0 / delta_t

    @property
    def delta_f(self):
        return 1.0 / self.time_duration

    @delta_f.setter
    def delta_f(self, delta_f):
        self.time_duration = 1.0 / delta_f

    @property
    def Nt(self):
        return int(self.time_duration * self.sampling_rate)

    @property
    def Nf(self):
        return int(self.f_max / self.delta_f) + 1

    @property
    def context_dim(self):
        if self.domain == 'TD':
            return self.Nt * len(self.detectors)
        elif self.domain == 'FD':
            return ((int((self.f_max - self.f_min) / self.delta_f) + 1)
                    * len(self.detectors) * 2)
        elif self.domain == 'RB':
            return self.Nrb * len(self.detectors) * 2

    @property
    def sample_times(self):
        """Array of times at which waveforms are sampled."""
        return np.linspace(0.0, self.time_duration,
                           num=self.Nt,
                           endpoint=False,
                           dtype=np.float32)

    @property
    @functools.lru_cache()
    def sample_frequencies(self):
        return np.linspace(0.0, self.f_max,
                           num=self.Nf, endpoint=True,
                           dtype=np.float32)

    @property
    @functools.lru_cache()
    def frequency_mask(self):
        return (self.sample_frequencies >= self.f_min)

    @property
    def _noise_std(self):
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.

        """
        if self.domain == 'TD':
            return 1.0 / np.sqrt(2.0 * self.delta_t)

        elif self.domain in ('FD', 'RB'):
            return np.sqrt(self.window_factor) / np.sqrt(4.0 * self.delta_f)

    @property
    def parameter_labels(self):
        labels = []
        for param in self.param_idx.keys():
            labels.append(self.parameters_latex_dict[param])
        return labels

    def init_detectors(self, ifo_list):
        """Create Detector objects.

        Arguments:
            ifo_list {list} -- list of strings representing detector names
        """

        # Instantiating the Detector objects is kind of slow.
        # Better to do it once, and save them for future use.
        self.detectors = {}
        for ifo in ifo_list:
            self.detectors[ifo] = Detector(ifo)

    #
    # Dataset generation
    #

    def generate_dataset(self, n=100000):
        """Generate and store the dataset of waveforms.

        Waveforms are distributed in parameter space according to the prior.

        Keyword Arguments:
            n {int} -- number of waveforms (default: {10000})
        """

        if self.domain == 'RB' and self.basis is None:
            self.generate_reduced_basis()

        print("Sampling {} sets of parameters from prior.".format(n))
        self.parameters = self._sample_prior(n)
        self.nsamples = len(self.parameters)

        if self.extrinsic_at_train:
            # Calculate h_+ and h_x, for reference PSD.
            # Only set up for FD or RB waveforms.

            if self.domain == 'TD':
                raise NotImplementedError('Cannot apply extrinsic '
                                          'parameters at train time in TD.')

            # Set extrinsic parameters to fiducial values.
            print("Setting extrinsic parameters to fiducial values.")
            for extrinsic_param, value in self.fiducial_params.items():
                self.parameters[:, self.param_idx[extrinsic_param]] = value

            print('Splitting parameters into training and test sets.')
            self.init_training()

            # Set up relative whitening
            self.init_relative_whitening()

            if self.domain == 'FD':
                wf_length = self.Nf
            elif self.domain == 'RB':
                wf_length = self.Nrb

            # Allocate storage
            self.hp = np.empty([n, wf_length], dtype=np.complex64)
            self.hc = np.empty([n, wf_length], dtype=np.complex64)

            # Generate waveforms
            print('Generating + and x waveforms.')
            # hp_FD = np.empty(self.Nf, dtype=np.complex64)
            # hc_FD = np.empty(self.Nf, dtype=np.complex64)
            for i in tqdm(range(n)):
                p = self.parameters[i]

                # FD waveforms
                hp, hc = self._generate_whitened_waveform(
                    p, intrinsic_only=True
                )
                hp = hp.astype(np.complex64)
                hc = hc.astype(np.complex64)

                if self.domain == 'RB':
                    # Convert FD to RB waveforms
                    hp = self.basis.fseries_to_basis_coefficients(hp)
                    hc = self.basis.fseries_to_basis_coefficients(hc)

                # Store waveforms
                self.hp[i] = hp
                self.hc[i] = hc

            # Additional initialization. Sets up whitening and time
            # translations.
            print('Performing additional dataset initialization.')
            if self.domain == 'FD':
                self.init_relative_whitening()
            elif self.domain == 'RB':
                self.initialize_reduced_basis_aux()

        else:
            # Calculate detector waveforms.
            # Only set up for TD or FD waveforms.

            if self.domain == 'RB':
                raise Exception('Not implemented. RB domain only works'
                                ' with train time extrinsic parameters.')

            for ifo in self.detectors.keys():
                if self.domain == 'TD':
                    self.h_detector[ifo] = np.empty(
                        [n, self.Nt], dtype=np.float32)
                elif self.domain == 'FD':
                    self.h_detector[ifo] = np.empty(
                        [n, self.Nf], dtype=np.complex64)

            for i, p in enumerate(self.parameters):
                h = self._generate_whitened_waveform(p)
                for ifo, wf in h.items():
                    self.h_detector[ifo][i] = wf

        # Cast parameters to float32 after generating the waveforms. We use
        # float32 / complex64 for all training data.

        self.parameters = self.parameters.astype(np.float32)

    def set_m1_m2_ranges_from_M_q_ranges(self):

        M_min, M_max = self.prior['M']
        q_min, q_max = self.prior['q']

        m1_min, _ = m1_m2_from_M_q(M_min, q_max)
        _, m2_min = m1_m2_from_M_q(M_min, q_min)
        m1_max, _ = m1_m2_from_M_q(M_max, q_min)
        _, m2_max = m1_m2_from_M_q(M_max, q_max)

        self.prior['mass_1'] = [m1_min, m1_max]
        self.prior['mass_2'] = [m2_min, m2_max]

    def _sample_prior(self, n):
        """Obtain samples from the prior distribution.

        Note that this does not respect the SNR threshold.

        Arguments:
            n {int} -- number of samples

        Returns:
            array -- samples
        """

        # Transform prior ranges to those of corresponding uniformly sampled
        # parameters
        uniform_prior = np.zeros((self.nparams, 2))
        for param, idx in self.param_idx.items():
            uniform_prior[idx] = self.prior[param]

            # Fix up parameters that are not already uniformly sampled
            if param in ('theta_jn', 'tilt_1', 'tilt_2'):
                uniform_prior[idx] = np.cos(uniform_prior[idx])
            elif param == 'dec':
                uniform_prior[idx] = np.sin(uniform_prior[idx])
            elif param == 'distance':
                uniform_prior[idx] = uniform_prior[idx] ** 3.0

        # Draw uniform samples
        draw = np.random.random((n, self.nparams))
        samples = np.apply_along_axis(lambda x: x*(uniform_prior[:, 1]
                                                   - uniform_prior[:, 0])
                                      + uniform_prior[:, 0], 1, draw)

        # If we have prior bounds on total mass and mass ratio, ensure that the
        # individual masses are such that these bounds are respected.
        #
        # Also apply m1 >= m2 convention

        m1i = self.param_idx['mass_1']
        m2i = self.param_idx['mass_2']

        if ('M' in self.prior.keys()) and ('q' in self.prior.keys()):
            M_min, M_max = self.prior['M']
            q_min, q_max = self.prior['q']
            m1_min, m1_max = self.prior['mass_1']
            m2_min, m2_max = self.prior['mass_2']
            for i in range(n):
                m1, m2 = samples[i, [m1i, m2i]]
                while True:
                    M, q = M_q_from_m1_m2(m1, m2)
                    if (m1 >= m2 and M >= M_min and M <= M_max
                            and q >= q_min and q <= q_max):
                        samples[i, [m1i, m2i]] = (m1, m2)
                        break
                    else:
                        m1 = m1_min + (m1_max - m1_min) * np.random.random()
                        m2 = m2_min + (m2_max - m2_min) * np.random.random()
        else:
            # ONLY VALID OF M1, M2 HAVE THE SAME RANGES
            samples[:, [m2i, m1i]] = np.sort(samples[:, [m1i, m2i]])

        # Undo uniformity transformations
        for param, idx in self.param_idx.items():
            if param in ('theta_jn', 'tilt_1', 'tilt_2'):
                samples[:, idx] = np.arccos(samples[:, idx])
            elif param == 'dec':
                samples[:, idx] = np.arcsin(samples[:, idx])
            elif param == 'distance':
                samples[:, idx] = samples[:, idx] ** (1.0/3.0)

        return samples

    def _generate_psd(self, delta_f, ifo):
        """Generate a PSD. This depends on the detector chosen.

        Arguments:
            delta_f {float} -- frequency spacing for PSD
            ifo {str} -- detector name

        Returns:
            psd -- generated PSD
        """

        # The PSD length should be the same as the length of FD
        # waveforms, which is determined from delta_f and f_max.

        psd_length = int(self.f_max / delta_f) + 1

        if self.event is None:
            psd = pycbc.psd.from_string(self.psd_names[ifo], psd_length,
                                        delta_f, self.f_min_psd)
        else:
            psd = pycbc.psd.from_txt(self.event_dir
                                     / (self.psd_names[ifo] + '.txt'),
                                     psd_length, delta_f, self.f_min_psd,
                                     is_asd_file=False)

        # To avoid division by 0 when whitening, set the PSD values
        # below f_min and for f_max to the boundary values.
        lower = int(self.f_min_psd / delta_f)
        psd[:lower] = psd[lower]
        psd[-1:] = psd[-2]

        return psd

    def _get_psd(self, delta_f, ifo):
        """Return a PSD with given delta_f.

        Either get the PSD from the PSD dictionary or generate it and
        save it to the PSD dictionary.

         Arguments:
            delta_f {float} -- frequency spacing for PSD
            ifo {str} -- detector name

        Returns:
            psd -- generated PSD
        """

        key = int(1.0/delta_f)

        if key not in self.psd[ifo]:
            self.psd[ifo][key] = self._generate_psd(delta_f, ifo)

        return self.psd[ifo][key]

    def _generate_whitened_waveform(self, p, intrinsic_only=False):
        """Return a whitened TD waveform generated with parameters p."""

        mass_1 = p[self.param_idx['mass_1']]
        mass_2 = p[self.param_idx['mass_2']]
        phase = p[self.param_idx['phase']]
        time = p[self.param_idx['time']]
        distance = p[self.param_idx['distance']]
        ra = p[self.param_idx['ra']]
        dec = p[self.param_idx['dec']]

        # Convert from source frame to Cartesian parameters
        # Optional parameters have default values

        if self.inclination:
            theta_jn = p[self.param_idx['theta_jn']]
            psi = p[self.param_idx['psi']]
        else:
            theta_jn = 0.0
            psi = 0.0

        if self.spins:
            if self.spins_aligned:
                spin_1x = 0.0
                spin_1y = 0.0
                spin_1z = p[self.param_idx['chi_1']]
                spin_2x = 0.0
                spin_2y = 0.0
                spin_2z = p[self.param_idx['chi_2']]
                iota = theta_jn
            else:
                a_1 = p[self.param_idx['a_1']]
                a_2 = p[self.param_idx['a_2']]
                tilt_1 = p[self.param_idx['tilt_1']]
                tilt_2 = p[self.param_idx['tilt_2']]
                phi_jl = p[self.param_idx['phi_jl']]
                phi_12 = p[self.param_idx['phi_12']]
                (iota, spin_1x, spin_1y, spin_1z,
                 spin_2x, spin_2y, spin_2z) = source_frame_to_radiation(
                     theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
                     mass_1, mass_2,
                     self.f_ref, phase)
        else:
            spin_1x = 0.0
            spin_1y = 0.0
            spin_1z = 0.0
            spin_2x = 0.0
            spin_2y = 0.0
            spin_2z = 0.0
            iota = theta_jn

        if self.domain == 'TD':
            # Start with a TD waveform generated from pycbc. If the
            # approximant is in FD, then this suitably tapers the low
            # frequencies in order to have a finite-length TD waveform
            # without wraparound effects. If we started with an FD
            # waveform, then we would have to do these manipulations
            # ourselves.

            # Make sure f_min is low enough
            if (self.time_duration >
                get_waveform_filter_length_in_time(mass1=mass_1, mass2=mass_2,
                                                   spin1x=spin_1x,
                                                   spin2x=spin_2x,
                                                   spin1y=spin_1y,
                                                   spin2y=spin_2y,
                                                   spin1z=spin_1z,
                                                   spin2z=spin_2z,
                                                   inclination=iota,
                                                   f_lower=self.f_min,
                                                   f_ref=self.f_ref,
                                                   approximant=self.approximant)):
                print('Warning: f_min not low enough for given '
                      'waveform duration')
                print(p)

            hp_TD, hc_TD = get_td_waveform(mass1=mass_1, mass2=mass_2,
                                           spin1x=spin_1x, spin2x=spin_2x,
                                           spin1y=spin_1y, spin2y=spin_2y,
                                           spin1z=spin_1z, spin2z=spin_2z,
                                           distance=distance,
                                           coa_phase=phase,
                                           inclination=iota,  # CHECK THIS!!!
                                           delta_t=self.delta_t,
                                           f_lower=self.f_min,
                                           f_ref=self.f_ref,
                                           approximant=self.approximant)
            hp = hp_TD.to_frequencyseries()
            hc = hc_TD.to_frequencyseries()

        elif self.domain in ('FD', 'RB'):
            if(is_fd_waveform(self.approximant)):
                # Use the pycbc waveform generator; change this later
                hp, hc = get_fd_waveform(mass1=mass_1, mass2=mass_2,
                                         spin1x=spin_1x, spin2x=spin_2x,
                                         spin1y=spin_1y, spin2y=spin_2y,
                                         spin1z=spin_1z, spin2z=spin_2z,
                                         distance=distance,
                                         coa_phase=phase,
                                         inclination=iota,
                                         f_lower=self.f_min,
                                         f_final=self.f_max,
                                         delta_f=self.delta_f,
                                         f_ref=self.f_ref,
                                         approximant=self.approximant)
            else:
                # Use SimInspiralFD. This converts automatically
                # from the TD to FD waveform, but it requires a timeshift to be
                # applied. Approach mimics bilby treatment.

                # Require SI units
                mass_1_SI = mass_1 * MSUN_SI
                mass_2_SI = mass_2 * MSUN_SI
                distance_SI = distance * PC_SI * 1e6

                lal_approximant = GetApproximantFromString(self.approximant)

                h_p, h_c = SimInspiralFD(mass_1_SI, mass_2_SI,
                                         spin_1x, spin_1y, spin_1z,
                                         spin_2x, spin_2y, spin_2z,
                                         distance_SI, iota, phase,
                                         0.0, 0.0, 0.0,
                                         self.delta_f, self.f_min, self.f_max,
                                         self.f_ref, None,
                                         lal_approximant)

                # If f_max/delta_f is not a power of 2, SimInspiralFD increases
                # f_max to make this a power of 2. Take only components running
                # up to f_max.
                hp = np.zeros_like(self.sample_frequencies,
                                   dtype=np.complex)
                hc = np.zeros_like(self.sample_frequencies,
                                   dtype=np.complex)
                hp[:] = h_p.data.data[:len(hp)]
                hc[:] = h_c.data.data[:len(hp)]

                # Zero the strain for frequencies below f_min
                hp *= self.frequency_mask
                hc *= self.frequency_mask

                # SimInspiralFD sets the merger time so the waveform can be
                # transformed to TD without wrapping the end of the waveform to
                # the beginning. Bring the time of coalescence to 0.
                dt = 1. / self.delta_f + (h_p.epoch.gpsSeconds +
                                          h_p.epoch.gpsNanoSeconds * 1e-9)
                hp *= np.exp(- 1j * 2 * np.pi * dt * self.sample_frequencies)
                hc *= np.exp(- 1j * 2 * np.pi * dt * self.sample_frequencies)

                # Convert to pycbc frequencyseries. Later, get rid of pycbc
                # functions.
                hp = FrequencySeries(hp, delta_f=self.delta_f,
                                     epoch=-self.time_duration)
                hc = FrequencySeries(hc, delta_f=self.delta_f,
                                     epoch=-self.time_duration)

        if intrinsic_only:
            # Whiten with reference noise PSD and return hp, hc

            hp = hp / (self._get_psd(hp.delta_f, 'ref') ** 0.5)
            hc = hc / (self._get_psd(hc.delta_f, 'ref') ** 0.5)

            # Convert to TD if necessary, ensure correct length
            if self.domain == 'TD':
                hp = hp.to_timeseries().time_slice(-self.time_duration, 0.0)
                hc = hc.to_timeseries().time_slice(-self.time_duration, 0.0)

            return hp.data, hc.data

        else:
            # Project waveform onto detectors

            h_d_dict = {}
            for ifo, d in self.detectors.items():

                # Project onto antenna pattern
                fp, fc = d.antenna_pattern(ra, dec, psi, self.ref_time)
                h_d = fp * hp + fc * hc

                # Apply time delay relative to Earth center
                dt = d.time_delay_from_earth_center(ra, dec, self.ref_time)
                time_d = time + dt

                # Merger is currently at time 0. Shift it.
                # NOT SURE NEXT LINE IS RIGHT / NEEDED. COMMENTED.
                # time_shift = - (self.time_duration - time_d)
                time_shift = time_d
                h_d = h_d.cyclic_time_shift(time_shift)
                h_d.start_time = h_d.start_time + time_shift

                # whiten
                h_d = h_d / (self._get_psd(h_d.delta_f, ifo) ** 0.5)

                # Convert to TD if necessary, and ensure waveform is of correct
                # length
                if self.domain == 'TD':
                    h_d = h_d.to_timeseries().time_slice(-self.time_duration,
                                                         0.0)

                h_d_dict[ifo] = h_d.data

            return h_d_dict

    #
    # Methods for train-time extrinsic parameters
    #

    def sample_prior_extrinsic(self, n):
        """Draw samples of extrinsic parameters from the prior.

        Arguments:
            n {int} -- number of prior samples

        Returns:
            array -- n x m array of samples, where m is number of extrinsic
                     parameters
        """

        nextrinsic = len(self.extrinsic_params)

        uniform_prior = np.zeros((nextrinsic, 2))
        for i, param in enumerate(self.extrinsic_params):
            uniform_prior[i] = self.prior[param]

            # Fix up non-uniformly distributed priors
            if param == 'dec':
                uniform_prior[i] = np.sin(uniform_prior[i])
            elif param == 'distance':
                uniform_prior[i] = uniform_prior[i] ** 3.0

        # Draw uniform samples
        draw = np.random.random((n, nextrinsic))
        samples = np.apply_along_axis(lambda x: x*(uniform_prior[:, 1]
                                                   - uniform_prior[:, 0])
                                      + uniform_prior[:, 0], 1, draw)

        # Undo uniformity transformations
        for i, param in enumerate(self.extrinsic_params):
            if param == 'dec':
                samples[:, i] = np.arcsin(samples[:, i])
            elif param == 'distance':
                samples[:, i] = samples[:, i] ** (1.0/3.0)

        return samples.astype(np.float32)

    def get_detector_waveforms(self, hp, hc, p_initial, p_extrinsic, mode):
        """Convert intrinsic hp, hc waveforms into waveforms at the detectors.

        This modifies the extrinsic parameters (distance, phase, time) and
        inserts the sky position.

        Works on FD or RB waveforms.

        Arguments:
            hp {array} -- plus polarization of initial waveform
            hc {array} -- cross polarization of initial waveform
            p_initial {array} -- parameters of initial waveform
            p_extrinsic {array} -- new extrinsic parameters desired
            mode {str} -- 'FD' or 'RB'

        Returns:
            tuple -- (new parameter array, list of detector waveforms)
        """

        if mode not in ('FD', 'RB'):
            raise Exception('Only works in FD or RB mode.')

        # Array of new parameter values
        p_new = p_initial.copy()
        for i, param in enumerate(self.extrinsic_params):
            p_new[self.param_idx[param]] = p_extrinsic[i]

        # Shifts in parameters relative to initial
        distance_scaling = (p_initial[self.param_idx['distance']]
                            / p_new[self.param_idx['distance']])
        time_shift_earth_center = (p_new[self.param_idx['time']]
                                   - p_initial[self.param_idx['time']])
        # phase_shift = (p_new[self.param_idx['phase']]
        #                - p_initial[self.param_idx['phase']])

        # Apply the phase and distance shifts to hp and hc
        #
        # Phase shift has a 2 because GW phase is twice orbital phase.
        #
        # DOES THE PHASE SHIFT NEED UPDATING FOR HIGHER MODES?
        scale = distance_scaling # * np.exp(2j * phase_shift)
        hp = hp * scale
        hc = hc * scale

        # Sky position parameters
        ra = p_new[self.param_idx['ra']]
        dec = p_new[self.param_idx['dec']]
        if self.inclination:
            psi = p_new[self.param_idx['psi']]
        else:
            psi = 0.0

        # Project onto detectors
        h_d_dict = {}
        for ifo, d in self.detectors.items():

            # Project onto antenna pattern
            fp, fc = d.antenna_pattern(ra, dec, psi, self.ref_time)
            h_d = fp * hp + fc * hc

            # Calculate time shift at detector
            dt = d.time_delay_from_earth_center(ra, dec, self.ref_time)
            time_shift = time_shift_earth_center + dt

            # Time translate and whiten
            if mode == 'FD':
                h_d = h_d * np.exp(- 2j * np.pi * time_shift
                                   * self.sample_frequencies)
                h_d = self.whiten_relative(h_d, ifo)

            elif mode == 'RB':
                h_d = self.basis.time_translate(h_d, time_shift,
                                                interpolation='cubic')
                h_d = self.basis.whiten(h_d, self.psd_names[ifo])

            # h_d_list.append(h_d)
            h_d_dict[ifo] = h_d

        return p_new, h_d_dict

    def init_relative_whitening(self):
        """Initialize relative whitening.

        For FD waveforms, this sets up multiplicative factors to go from
        waveforms whitened with the reference PSD to waveforms
        whitened with detector PSDs.
        """

        ref_psd_name = self.psd_names['ref']
        ref_psd = np.array(self._get_psd(self.delta_f, 'ref'))

        self.relative_whitening_dict = {}
        for ifo in self.detectors.keys():
            psd_name = self.psd_names[ifo]
            if ((psd_name != ref_psd_name)
                    and (psd_name not in self.relative_whitening_dict.keys())):
                psd = np.array(self._get_psd(self.delta_f, ifo))

                # Multiply FD waveform by this factor to whiten
                whitening_factor = (ref_psd / psd) ** 0.5
                whitening_factor = whitening_factor.astype(np.float32)

                self.relative_whitening_dict[psd_name] = whitening_factor

    def whiten_relative(self, h, ifo):
        """Whiten a FD waveform that has already been whitened with the
        reference PSD.

        Whitening must first be initialized with init_relative_whitening.

        Arguments:
            h {array} -- frequency domain waveform
            ifo {str} -- detector name for whitening

        Returns:
            array -- whitened waveform
        """

        if self.psd_names[ifo] != self.psd_names['ref']:
            return h * self.relative_whitening_dict[self.psd_names[ifo]]
        else:
            return h

    def p_h_random_extrinsic(self, idx, train, mode=None):
        """Generate detector waveform with random extrinsic parameters.

        This uses intrinsic parameters for a given index from either the train
        or test set. If necessary, it generates the + and x polarizations.

        Then it generates random parameters, and calculates detector waveforms.

        Arguments: idx {int} -- index of the intrinsic parameters train {bool}
            -- True: training set; False: test set

        Keyword Arguments: mode {str} -- domain of desired waveform ('FD' or
            'RB') (default: {None})

        Returns: (array, dict, float) -- parameters, detector waveforms, weight
        """

        if mode is None:
            mode = self.domain

        if mode not in ('FD', 'RB'):
            raise Exception('Method only implemented in FD or RB.')

        # Translate idx to an index of the full dataset
        if train:
            orig_idx = self.train_selection[idx]
        else:
            orig_idx = self.test_selection[idx]

        # Parameters

        # Take intrinsic parameters corresponding to idx.
        p_initial = self.parameters[orig_idx]

        # Generate random extrinsic parameters.
        p_extrinsic = self.sample_prior_extrinsic(1)[0]

        if mode == 'FD' and self.domain == 'RB':
            # Generate the waveform.
            #
            # This usually only gets run when generating noisy test FD
            # waveforms for the RB network. The waveforms are already
            # saved in RB, but we require FD waveforms.

            # Intrinsic waveform
            #
            # Type conversion of the parameters is needed for lal.
            hp, hc = self._generate_whitened_waveform(
                p_initial.astype(np.float64),
                intrinsic_only=True)
            hp = hp.astype(np.complex64)
            hc = hc.astype(np.complex64)

        else:
            # Waveforms are already saved in dataset.
            hp = self.hp[orig_idx]
            hc = self.hc[orig_idx]

        # Apply extrinsic parameters to obtain detector waveforms
        p, h_det = self.get_detector_waveforms(hp, hc, p_initial, p_extrinsic,
                                               mode)

        # If we have an SNR threshold, resample the distance such that the
        # threshold is respected.
        if ((self.snr_threshold is not None)
                or (self.distance_prior_fn is not None)):
            p, h_det, weight = self._resample_distance(p, h_det)
        else:
            weight = 1.0

        stacked = np.hstack(list(h_det.values()))
        snr = (np.sqrt(np.sum(np.abs(stacked)**2)) / self._noise_std)

        return p, h_det, weight, snr

    #
    # Methods for reduced basis
    #

    def generate_reduced_basis(self, n_train=10000, n_test=10000):
        """Generate the reduced basis elements.

        This draws parameters from the prior, generates detector waveforms,
        and trains the SVD basis based on these.

        It then evaluates performance on the training waveforms, and a set
        of validation waveforms.

        Keyword Arguments:
            n_train {int} -- number of training waveforms (default: {10000})
            n_test {int} -- number of test waveforms (default: {10000})
        """

        print('Generating {} detector FD waveforms for training reduced basis.'
              .format(n_train))

        h_detector = {}
        for ifo in self.detectors.keys():
            h_detector[ifo] = np.empty((n_train, self.Nf), dtype=np.complex64)

        for i in tqdm(range(n_train)):
            p = self._sample_prior(1)[0]
            # To generate reduced basis, fix all waveforms to same fiducial
            # distance.
            p[self.param_idx['distance']] = self.fiducial_params['distance']
            h_d = self._generate_whitened_waveform(p, intrinsic_only=False)
            for ifo, h in h_d.items():
                h_detector[ifo][i] = h

        print('Generating reduced basis for training detector waveforms')

        training_array = np.vstack(list(h_detector.values()))
        self.basis = SVDBasis()
        self.basis.generate_basis(training_array, n=self.Nrb)

        # print('Calculating standard deviations for training standardization.')
        # for ifo, h_array_FD in h_detector.items():

        #     # Project training data for given ifo onto reduced basis
        #     h_array_RB = np.empty((n_train, self.Nrb), dtype=np.complex64)
        #     for i, h in enumerate(h_array_FD):
        #         h_array_RB[i] = self.basis.fseries_to_basis_coefficients(h)

        #     # Compute standardization for given ifo
        #     self.basis.init_standardization(ifo, h_array_RB, self._noise_std)

        print('Evaluating performance on training set waveforms.')
        matches = []
        for h_FD in tqdm(training_array):
            h_RB = self.basis.fseries_to_basis_coefficients(h_FD)
            h_reconstructed = self.basis.basis_coefficients_to_fseries(h_RB)

            norm1 = np.mean(np.abs(h_FD)**2)
            norm2 = np.mean(np.abs(h_reconstructed)**2)
            inner = np.mean(h_FD.conj()*h_reconstructed).real

            matches.append(inner / np.sqrt(norm1 * norm2))
        mismatches = 1 - np.array(matches)
        print('  Mean mismatch = {}'.format(np.mean(mismatches)))
        print('  Standard deviation = {}'.format(np.std(mismatches)))
        print('  Max mismatch = {}'.format(np.max(mismatches)))
        print('  Median mismatch = {}'.format(np.median(mismatches)))
        print('  Percentiles:')
        print('    99    -> {}'.format(np.percentile(mismatches, 99)))
        print('    99.9  -> {}'.format(np.percentile(mismatches, 99.9)))
        print('    99.99 -> {}'.format(np.percentile(mismatches, 99.99)))

        # Evaluation on test waveforms

        print('Generating {} detector FD waveforms for testing reduced basis.'
              .format(n_test))

        h_detector = {}
        for ifo in self.detectors.keys():
            h_detector[ifo] = np.empty((n_test, self.Nf), dtype=np.complex64)

        for i in tqdm(range(n_test)):
            p = self._sample_prior(1)[0]
            # To generate reduced basis, fix all waveforms to same fiducial
            # distance.
            p[self.param_idx['distance']] = self.fiducial_params['distance']
            h_d = self._generate_whitened_waveform(p, intrinsic_only=False)
            for ifo, h in h_d.items():
                h_detector[ifo][i] = h

        print('Evaluating performance on test set waveforms.')
        test_array = np.vstack(list(h_detector.values()))
        matches = []
        for h_FD in tqdm(test_array):
            h_RB = self.basis.fseries_to_basis_coefficients(h_FD)
            h_reconstructed = self.basis.basis_coefficients_to_fseries(h_RB)

            norm1 = np.mean(np.abs(h_FD)**2)
            norm2 = np.mean(np.abs(h_reconstructed)**2)
            inner = np.mean(h_FD.conj()*h_reconstructed).real

            matches.append(inner / np.sqrt(norm1 * norm2))
        mismatches = 1 - np.array(matches)
        print('  Mean mismatch = {}'.format(np.mean(mismatches)))
        print('  Standard deviation = {}'.format(np.std(mismatches)))
        print('  Max mismatch = {}'.format(np.max(mismatches)))
        print('  Median mismatch = {}'.format(np.median(mismatches)))
        print('  Percentiles:')
        print('    99    -> {}'.format(np.percentile(mismatches, 99)))
        print('    99.9  -> {}'.format(np.percentile(mismatches, 99.9)))
        print('    99.99 -> {}'.format(np.percentile(mismatches, 99.99)))

        # print('Projecting plus and cross polarizations onto basis.')

        # self.hp = np.empty((n, self.Nrb), dtype=np.complex64)
        # self.hc = np.empty((n, self.Nrb), dtype=np.complex64)
        # for i in tqdm(range(n)):
        #     hp_RB = self.basis.fseries_to_basis_coefficients(self.hp_FD[i])
        #     hc_RB = self.basis.fseries_to_basis_coefficients(self.hc_FD[i])

        #     self.hp[i] = hp_RB
        #     self.hc[i] = hc_RB

    def initialize_reduced_basis_aux(self):
        """Initialize the reduced basis to be able to perform time translation
        and relative whitening transformations.
        """

        # Relative whitening

        ref_psd_name = self.psd_names['ref']
        ref_psd = np.array(self._get_psd(self.delta_f, 'ref'))

        for ifo, psd_name in self.psd_names.items():
            psd = np.array(self._get_psd(self.delta_f, ifo))

            self.basis.init_whitening(ref_psd_name, ref_psd, psd_name, psd)

        # Time translations

        # Add the earth-crossing time to the ends of the time prior.
        # This is overkill, since we really just need the radius.
        earth_crossing_time = 2 * REARTH_SI / C_SI
        t_min = self.prior['time'][0] - earth_crossing_time
        t_max = self.prior['time'][1] + earth_crossing_time

        self.basis.init_time_translation(t_min, t_max, TIME_TRANSLATION_PTS,
                                         self.sample_frequencies)

    def truncate_basis(self, n):
        """Truncate the reduced basis to dimension n.

        Arguments:
            n {int} -- New basis dimension.
        """

        if n >= self.Nrb:
            print('Reduced basis has {} components already.'.format(self.Nrb))
            return
        else:
            print('Truncating reduced basis from {} to {} elements.'.format(
                self.Nrb, n
            ))

        self.basis.truncate(n)
        self.Nrb = n

        self.hp = self.hp[:, :n]
        self.hc = self.hc[:, :n]

    #
    # File I/O for waveform database
    #

    def save(self, data_dir='.', data_fn='waveform_dataset.hdf5',
             config_fn='settings.json'):
        """Save the database of parameters and waveforms to an HDF5 file,
        and the configuration parameters to a json file.

        Keyword Arguments:
            data_dir {str} -- directory for saving (default: {'.'})
            data_fn {str} -- data file name (default:
                                             {'waveform_dataset.hdf5'})
            config_fn {str} -- configuration file name (default:
                                             {'settings.json'})
        """
        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)

        # Save configuration parameters

        with open(p / config_fn, 'w') as f_config:
            json.dump(dict(domain=self.domain,
                           prior=self.prior,
                           approximant=self.approximant,
                           params=self.param_idx,
                           latex=self.parameters_latex_dict,
                           detectors=list(self.detectors.keys()),
                           psds=self.psd_names,
                           f_min=self.f_min,
                           f_min_psd=self.f_min_psd,
                           sampling_rate=self.sampling_rate,
                           time_duration=self.time_duration,
                           ref_time=self.ref_time,
                           f_ref=self.f_ref,
                           extrinsic_at_train=self.extrinsic_at_train,
                           extrinsic_params=self.extrinsic_params,
                           fiducial_params=self.fiducial_params,
                           window_factor=self.window_factor,
                           event=self.event,
                           event_dir=str(self.event_dir)
                           ), f_config, indent=4)

        # Save data (waveforms, parameters)

        f_data = h5py.File(p / data_fn, 'w')

        f_data.create_dataset('parameters', data=self.parameters,
                              compression='gzip', compression_opts=9)

        if self.extrinsic_at_train:
            # Save plus and cross polarizations
            f_data.create_dataset('hp', data=self.hp,
                                  compression='gzip', compression_opts=9)
            f_data.create_dataset('hc', data=self.hc,
                                  compression='gzip', compression_opts=9)
        else:
            # Save detector waveforms
            hgroup = f_data.create_group('h')
            for detector, h in self.h_detector.items():
                hgroup.create_dataset(detector, data=h,
                                      compression='gzip', compression_opts=9)

        f_data.close()

        if self.domain == 'RB':
            self.basis.save(data_dir)

    def load(self, data_dir='.', data_fn='waveform_dataset.hdf5',
             config_fn='settings.json'):
        """Load a database created with the save method.

        Keyword Arguments:
            data_dir {str} -- directory where files are stored (default: {'.'})
            data_fn {str} -- data file name (default:
                                             {'waveform_dataset.hdf5'})
            config_fn {str} -- configuration file name (default:
                                                        {'settings.json'})
        """

        p = Path(data_dir)

        # Load configuration

        with open(p / config_fn, 'r') as f_config:
            d = json.load(f_config)
            self.prior = d['prior']
            self.approximant = d['approximant']
            self.param_idx = d['params']
            self.parameters_latex_dict = d['latex']
            ifos = d['detectors']
            self.init_detectors(ifos)
            self.psd_names = d['psds']
            self.f_min = d['f_min']
            self.f_min_psd = d['f_min_psd']
            self.sampling_rate = d['sampling_rate']
            self.time_duration = d['time_duration']
            self.ref_time = d['ref_time']
            if 'extrinsic_at_train' in d.keys():  # Compatibility
                self.extrinsic_at_train = d['extrinsic_at_train']
            else:
                self.extrinsic_at_train = False
            if 'extrinsic_params' in d.keys():
                self.extrinsic_params = d['extrinsic_params']
            if 'fiducial_params' in d.keys():
                self.fiducial_params = d['fiducial_params']
            if 'f_ref' in d.keys():
                self.f_ref = d['f_ref']
            if 'domain' in d.keys():  # Compatibility
                self.domain = d['domain']
            else:
                self.domain = 'TD'
            try:
                self.event = d['event']
                event_dir = d['event_dir']
                if event_dir != 'None':
                    self.event_dir = Path(event_dir)
                self.load_event(self.event_dir)
            except:
                self.event = None
                self.event_dir = None

        self.nparams = len(self.param_idx)

        self.spins = False
        self.inclination = False
        if (('chi_1' in self.param_idx.keys()) or
                ('chi1z') in self.param_idx.keys()):
            self.spins = True
            self.spins_aligned = True
        if 'a_1' in self.param_idx.keys():
            self.spins = True
            self.spins_aligned = False
        if (('theta_jn' in self.param_idx.keys())
                or ('inc' in self.param_idx.keys())):
            self.inclination = True

        # Load data

        f_data = h5py.File(p / data_fn, 'r')

        self.parameters = f_data['parameters'][:, :]
        self.nsamples = len(self.parameters)

        self.h_detector = {}

        if self.extrinsic_at_train:
            self.hp = f_data['hp'][:, :]
            self.hc = f_data['hc'][:, :]

            if self.domain == 'FD':
                self.init_relative_whitening()

        else:
            if 'h' in f_data.keys():
                hgroup = f_data['h']
            else:
                hgroup = f_data['h_whitened_TD']  # Compatibility
            for ifo in self.detectors.keys():
                self.h_detector[ifo] = hgroup[ifo][:, :]

        f_data.close()

        if self.domain == 'RB':
            self.basis = SVDBasis()
            self.basis.load(data_dir)
            self.Nrb = self.basis.n
            # self.initialize_reduced_basis_aux()

    #
    # Methods for working with training / test sets
    #

    def init_training(self, train_fraction=0.9):
        """Define training and test sets, compute parameters needed for
        standardization.

        """

        self.train_fraction = train_fraction

        # Define train and test sets
        ntrain = int(round(train_fraction * self.nsamples))
        self.train_selection = np.random.choice(range(self.nsamples),
                                                size=ntrain,
                                                replace=False)
        self.test_selection = np.array([i for i in range(self.nsamples)
                                        if i not in self.train_selection])

        self._compute_parameter_statistics()

    def _compute_parameter_statistics(self):
        """Compute mean and standard deviation for physical parameters, in
        order to standardize later.

        """
        # parameters_train = self.parameters[self.train_selection]
        # self.parameters_mean = np.mean(parameters_train, axis=0)
        # self.parameters_std = np.std(parameters_train, axis=0)

        self.parameters_mean = np.empty(self.nparams, dtype=np.float32)
        self.parameters_std = np.empty(self.nparams, dtype=np.float32)

        # Use analytic expressions

        for param, i in self.param_idx.items():
            left, right = self.prior[param]

            if param == 'mass_1':
                m2left, m2right = self.prior['mass_2']
                mean = ((-3*m2left*(left + right)
                         + 2*(left**2 + left*right + right**2))
                        / (3.*(left - 2*m2left + right)))
                cov = (((left - right)**2*(left**2 + 6*m2left**2
                                           + 4*left*right + right**2
                                           - 6*m2left*(left + right)))
                       / (18.*(left - 2*m2left + right)**2))
                std = np.sqrt(cov)

            elif param == 'mass_2':
                m1left, m1right = self.prior['mass_1']
                mean = ((-3*left**2 + m1left**2 + m1left*m1right + m1right**2)
                        / (3.*(-2*left + m1left + m1right)))
                cov = ((-2*(-3*left**2 + m1left**2
                            + m1left*m1right + m1right**2)**2 +
                        3*(-2*left + m1left + m1right) *
                        (-4*left**3
                         + (m1left + m1right)*(m1left**2 + m1right**2))) /
                       (18.*(-2*left + m1left + m1right)**2))
                std = np.sqrt(cov)

            if param in ('phase', 'time',
                         'chi_1', 'chi_2', 'a_1', 'a_2',
                         'phi_12', 'phi_jk', 'psi', 'ra'):
                # Uniform prior
                mean = (left + right)/2
                std = np.sqrt(((left - right)**2) / 12)

            elif param == 'distance':
                # Uniform in distance^3
                mean = ((3/4) * (left + right) * (left**2 + right**2)
                        / (left**2 + left*right + right**2))
                std = np.sqrt((3*((left - right)**2)
                               * (left**4 + 4*(left**3)*right
                                  + 10*(left**2)*(right**2)
                                  + 4*left*(right**3) + right**4))
                              / (80.*((left**2 + left*right + right**2)**2)))

            elif param in ('tilt_1', 'tilt_2', 'theta_jn'):
                # Uniform in cosine prior
                # Assume range is [0, pi]
                mean = np.pi / 2.0
                std = np.sqrt((np.pi**2 - 8) / 4)

            elif param == 'dec':
                # Uniform in sine prior
                # Assume range for declination is [-pi/2, pi/2]
                mean = 0.0
                std = np.sqrt((np.pi**2 - 8) / 4)

            self.parameters_mean[i] = mean
            self.parameters_std[i] = std

    def x_train(self):
        """Return training set of standardized waveform parameters x.

        Each parameter should have mean 0 and standard deviation 1,
        when averaged over the training set."""

        return (self.parameters[self.train_selection]
                - self.parameters_mean) / self.parameters_std

    def x_test(self):
        """Return test set of standardized waveform parameters x.

        Each parameter should have mean 0 and standard deviation 1,
        when averaged over the training set.

        """

        return (self.parameters[self.test_selection]
                - self.parameters_mean) / self.parameters_std

    def post_process_parameters(self, parameters):
        """Takes as input an array of size (nsamples, nparameters), consisting
        of a list of standardized parameters.

        Returns true parameters, i.e., undoes the standardization.
        """

        if parameters.shape[-1] != self.nparams:
            raise Exception('Error: wrong number of parameters.')

        return parameters * self.parameters_std + self.parameters_mean

    def pre_process_parameters(self, parameters):
        """Takes as input an array of size (nsamples, nparameters), consisting
        of a list of parameters.

        Returns standardized parameters.
        """

        if parameters.shape[-1] != self.nparams:
            raise Exception('Error: wrong number of parameters.')

        return (parameters - self.parameters_mean) / self.parameters_std

    def h_train(self):
        """Return training set of standardized whitened waveforms.
        """

        if self.domain == 'TD':
            h_joined = np.concatenate([self.h_detector[d]
                                       for d in self.detectors.keys()],
                                      axis=-1)

        elif self.domain == 'FD':

            # Cut out the part of the waveforms below f_min
            start_idx = int(self.f_min / self.delta_f)

            h_joined = []
            for d in self.detectors.keys():
                h_truncated = self.h_detector[d][:, start_idx:]
                h_joined.append(h_truncated.real)
                h_joined.append(h_truncated.imag)
            h_joined = np.concatenate(h_joined, axis=-1)

        return h_joined[self.train_selection]/self._noise_std

    def h_test(self):
        """Return test set of standardized whitened waveforms.
        """

        if self.domain == 'TD':
            h_joined = np.concatenate([self.h_detector[d]
                                       for d in self.detectors.keys()],
                                      axis=-1)

        elif self.domain == 'FD':

            # Cut out the part of the waveforms below f_min
            start_idx = int(self.f_min / self.delta_f)

            h_joined = []
            for d in self.detectors.keys():
                h_truncated = self.h_detector[d][:, start_idx:]
                h_joined.append(h_truncated.real)
                h_joined.append(h_truncated.imag)
            h_joined = np.concatenate(h_joined, axis=-1)

        return h_joined[self.test_selection]/self._noise_std

    def x_y_from_p_h(self, p, h, add_noise):

        if self.domain == 'RB':
            n = self.Nrb
        elif self.domain == 'FD':
            n = int(self.f_max / self.delta_f) + 1
        elif self.domain == 'TD':
            n = self.Nt

        # Standardize parameters
        x = (p - self.parameters_mean) / self.parameters_std

        # Repackage detector waveforms and add (optionally) add noise
        y_list = []
        for ifo, d in h.items():

            # Add noise. Waveforms are assumed to be white in each detector.

            if add_noise:

                if self.domain in ('RB', 'FD'):
                    noise = (np.random.normal(scale=self._noise_std, size=n)
                             + np.random.normal(scale=self._noise_std, size=n)*1j)
                    noise = noise.astype(np.complex64)

                elif self.domain == 'TD':
                    noise = np.random.normal(scale=self._noise_std, size=n)
                    noise = noise.astype(np.float32)

                d = d + noise

            # Standardize.

            if self.domain == 'RB':
                d = self.basis.standardize(d, ifo)

            elif self.domain in ('FD', 'TD'):
                d = d / self._noise_std

            # Repackage.

            if self.domain == 'FD':
                # Remove components below f_min
                start_idx = int(self.f_min / self.delta_f)
                d = d[start_idx:]

            if self.domain in ('FD', 'RB'):
                # Real and imaginary parts separately
                y_list.append(d.real)
                y_list.append(d.imag)

            else:
                y_list.append(d)

        y = np.hstack(y_list)

        return x, y

    def save_train(self, data_dir='.', filename='traintest_split.hdf5'):
        """Save the list of training and test elements."""

        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        f = h5py.File(p / filename, 'w')

        f.attrs['train_fraction'] = self.train_fraction
        f.create_dataset('train_selection', data=self.train_selection)
        f.create_dataset('test_selection', data=self.test_selection)

        f.close()

    def load_train(self, data_dir='.', filename='traintest_split.hdf5'):
        """Load the list of training and test elements, compute necessary
        statistics for standardization."""

        p = Path(data_dir)
        f = h5py.File(p / filename, 'r')

        self.train_fraction = f.attrs['train_fraction']
        self.train_selection = f['train_selection'][:]
        self.test_selection = f['test_selection'][:]

        f.close()

        self._compute_parameter_statistics()

    #
    # Utilities for working with waveforms with noise.
    #
    # During training, noise should be added by the training loop. These
    # utilities are for working with a standardized set of noisy waveforms in
    # order to compare parameter posteriors generated using different methods.
    #

    def generate_noisy_test_data(self, n=None):
        """Add unit gaussian noise to each standardized waveform in the test
         set."""

        if self.test_selection is None:
            raise NameError("No test set defined.")

        if n is None:
            n = len(self.test_selection)

        # Create parameter and waveform arrays

        self.noisy_waveforms_parameters = np.empty((n, self.nparams),
                                                   dtype=np.float32)

        self.noisy_test_waveforms = {}

        print('Generating white noise.')
        for ifo in self.detectors.keys():

            # Start with pure noise

            if self.domain in ('RB', 'FD'):
                noise = (np.random.normal(scale=self._noise_std,
                                          size=(n, self.Nf))
                         + np.random.normal(scale=self._noise_std,
                                            size=(n, self.Nf)) * 1j)
                noise = noise.astype(np.complex64)

            elif self.domain == 'TD':
                noise = np.random.normal(scale=self._noise_std,
                                         size=(n, self.Nt))
                noise = noise.astype(np.float32)

            self.noisy_test_waveforms[ifo] = noise

        print('Generating whitened detector waveforms.')
        for i in tqdm(range(n)):

            # Add in the waveform

            if self.extrinsic_at_train:
                p, h, _, _ = self.p_h_random_extrinsic(i, train=False, mode='FD')
                self.noisy_waveforms_parameters[i] = p
                for ifo, h_array in self.noisy_test_waveforms.items():
                    h_array[i] += h[ifo]

            else:
                orig_idx = self.test_selection[i]
                p = self.parameters[orig_idx]
                self.noisy_waveforms_parameters[i] = p
                for ifo, h_array in self.h_detector.items():
                    h = h_array[orig_idx]
                    self.noisy_test_waveforms[ifo][i] += h

        # else:
        #     h = self.h_test()
        #     noise = np.random.standard_normal(h.shape).astype(np.float32)
        #     self.noisy_test_waveforms = h + noise
        #     self.noisy_waveforms_parameters = self.parameters[self.test_selection]

    def save_noisy_test_data(self, data_dir='.',
                             filename='noisy_test_data.hdf5'):
        """Save noisy test data to file."""

        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        f = h5py.File(p / filename, 'w')

        if self.noisy_test_waveforms is None:
            self.generate_noisy_test_data()

        # f.create_dataset('noisy_waveforms', data=self.noisy_test_waveforms)
        hgroup = f.create_group('injections')
        for ifo, h_array in self.noisy_test_waveforms.items():
            hgroup.create_dataset(ifo, data=h_array,
                                  compression='gzip', compression_opts=9)

        # hgroup = f.create_group('clean')
        # for ifo, h_array in self.noisy_test_waveforms.items():
        #    hgroup.create_dataset(ifo, data=h_array,
        #                          compression='gzip', compression_opts=9)

        f.create_dataset('parameters', data=self.noisy_waveforms_parameters)
        f.create_dataset('parameters_mean', data=self.parameters_mean)
        f.create_dataset('parameters_std', data=self.parameters_std)

        f.close()

    def load_noisy_test_data(self, data_dir='.',
                             data_fn='noisy_test_data.hdf5',
                             config_fn='settings.json'):
        """Load noisy test data from file.

        This works even without loading a full training set, e.g, for
        evaluation purposes.
        """

        p = Path(data_dir)

        # Load configuration

        with open(p / config_fn, 'r') as f_config:
            d = json.load(f_config)
            self.prior = d['prior']
            self.approximant = d['approximant']
            self.param_idx = d['params']
            self.parameters_latex_dict = d['latex']
            ifos = d['detectors']
            self.init_detectors(ifos)
            self.psd_names = d['psds']
            self.f_min = d['f_min']
            self.f_min_psd = d['f_min_psd']
            self.sampling_rate = d['sampling_rate']
            self.time_duration = d['time_duration']
            self.ref_time = d['ref_time']
            if 'extrinsic_at_train' in d.keys():  # Compatibility
                self.extrinsic_at_train = d['extrinsic_at_train']
            else:
                self.extrinsic_at_train = False
            if 'extrinsic_params' in d.keys():
                self.extrinsic_params = d['extrinsic_params']
            if 'fiducial_params' in d.keys():
                self.fiducial_params = d['fiducial_params']
            if 'f_ref' in d.keys():
                self.f_ref = d['f_ref']
            if 'domain' in d.keys():
                self.domain = d['domain']
            else:
                self.domain = 'TD'

        self.nparams = len(self.param_idx)

        self.spins = False
        self.inclination = False
        if (('chi_1' in self.param_idx.keys()) or
                ('chi1z') in self.param_idx.keys()):
            self.spins = True
            self.spins_aligned = True
        if 'a_1' in self.param_idx.keys():
            self.spins = True
            self.spins_aligned = False
        if (('theta_jn' in self.param_idx.keys())
                or ('inc' in self.param_idx.keys())):
            self.inclination = True

        # Load noisy waveforms. It is also necessary to have the
        # standardization parameters used for training.

        f_data = h5py.File(p / data_fn, 'r')

        if 'noisy_waveforms' in f_data.keys():
            # For compatibility with old format
            self.noisy_test_waveforms = f_data['noisy_waveforms'][:, :]
        else:
            hgroup = f_data['injections']
            self.noisy_test_waveforms = {}
            for ifo in hgroup.keys():
                self.noisy_test_waveforms[ifo] = hgroup[ifo][:, :]

        self.noisy_waveforms_parameters = f_data['parameters'][:, :]

        self.parameters_mean = f_data['parameters_mean'][:]
        self.parameters_std = f_data['parameters_std'][:]

        f_data.close()

        if self.domain == 'RB':
            self.basis = SVDBasis()
            self.basis.load(data_dir)
            self.Nrb = self.basis.n
            # self.initialize_reduced_basis_aux()

    #
    # Real data
    #

    def load_event(self, event_dir):

        p = Path(event_dir)
        self.event_dir = p

        # Load event info
        with open(p / 'event_info.json', 'r') as f:
            d = json.load(f)
            self.event = d['event']
            self.f_min = d['f_min']
            self.f_min_psd = self.f_min
            self.f_max = d['f_max']
            self.time_duration = d['T']
            self.ref_time = d['t_event']
            self.window_factor = d['window_factor']
            detectors = d['detectors']

        # Initialize detectors
        self.init_detectors(detectors)

        # Set up PSD
        self.psd = {}
        self.psd_names = {}
        for ifo in detectors:
            self.psd[ifo] = {}
            self.psd_names[ifo] = 'PSD_{}'.format(ifo)
        self.psd['ref'] = {}
        self.psd_names['ref'] = self.psd_names[detectors[0]]

    #
    # Methods for working with SNR threshold / changing distance prior
    #

    def _resample_distance(self, p, h_det):
        """Resample the luminosity distance for a waveform based on
        new distance prior and / or an SNR threshold.

        Arguments:
            p {array} -- initial parameters
            h_det {dict} -- initial detector waveforms

        Returns:
            array -- new parameters, with distance resampled
            dict -- new detector waveforms
            float -- weight factor, (new volume) / (prior volume)
        """

        # SNR and luminosity distance of signal

        snr = (np.sqrt(np.sum(np.abs(np.hstack(list(h_det.values())))**2))
               / self._noise_std)
        distance = p[self.param_idx['distance']]

        if self.snr_threshold is not None:
            # New distance range. We are being a bit conservative, by ensuring
            # the range is at least some minimum distance beyond the lower
            # bound of the prior range.

            threshold_distance = distance * snr / self.snr_threshold
            lower_bound = self.prior['distance'][0]
            if threshold_distance > self.prior['distance'][1]:
                upper_bound = self.prior['distance'][1]
            elif threshold_distance > lower_bound + self.distance_buffer:
                upper_bound = threshold_distance
            else:
                upper_bound = lower_bound + self.distance_buffer

        else:
            lower_bound, upper_bound = self.prior['distance']

        # Sample a new distance

        if self.distance_prior_fn == 'uniform_distance':
            # Use a uniform-in-luminosity distance distribution q.

            new_distance = (lower_bound
                            + np.random.random() * (upper_bound - lower_bound))
            q_prob = 1 / (upper_bound - lower_bound)

        elif self.distance_prior_fn == 'inverse_distance':
            # Take q ~ 1/d_L

            # Sample from uniform in [0,1]
            u = np.random.random()

            # Inverse CDF(q)
            new_distance = lower_bound * np.exp(
                (np.log(upper_bound) - np.log(lower_bound)) * u)

            # evaluate q(d_L)
            q_prob = 1 / ((np.log(upper_bound) - np.log(lower_bound))
                          * new_distance)

        elif self.distance_prior_fn == 'inverse_square_distance':
            # q ~ 1/d_L^2

            # Sample from uniform in [0,1]
            u = np.random.random()

            # FIX

            # Inverse CDF(q)
            new_distance = (lower_bound * upper_bound /
                            (upper_bound - u * (upper_bound - lower_bound)))

            # q(d_L)
            q_prob = (upper_bound * lower_bound /
                      (new_distance**2 * (upper_bound - lower_bound)))

        elif self.distance_prior_fn == 'linear_distance':
            # Take q ~ d_L

            # Sample from uniform in [0,1]
            u = np.random.random()

            # Inverse CDF(q)
            new_distance = (lower_bound**2
                            + u * (upper_bound**2 - lower_bound**2))**(1/2)

            # q(d_L)
            q_prob = 2*new_distance / (upper_bound**2 - lower_bound**2)

        elif self.distance_prior_fn == 'power_distance':
            # Take q ~ d_L^\alpha

            # Sample from uniform in [0,1]
            u = np.random.random()

            if self.distance_power == -1.0:
                # Inverse CDF(q)
                new_distance = lower_bound * np.exp(
                    (np.log(upper_bound) - np.log(lower_bound)) * u)

                # evaluate q(d_L)
                q_prob = 1 / ((np.log(upper_bound) - np.log(lower_bound))
                              * new_distance)

            else:
                # Inverse CDF(q)
                a = self.distance_power + 1.0
                new_distance = (lower_bound**a
                                + u * (upper_bound**a - lower_bound**a))**(1/a)

                # q(d_L)
                q_prob = (a*new_distance**(a-1) /
                          (upper_bound**a - lower_bound**a))

        elif self.distance_prior_fn == 'bayeswave':
            u = np.random.random()
            new_distance = bw_inverse_cdf(u, self.bw_dstar,
                                          lower_bound, upper_bound)
            q_prob = bw_pdf(new_distance, self.bw_dstar,
                            lower_bound, upper_bound)

        else:
            # Use a volumetric prior

            new_vol = (lower_bound**3
                       + np.random.random() * (upper_bound**3
                                               - lower_bound**3))
            new_distance = new_vol ** (1/3)
            q_prob = 3 * new_distance**2 / (upper_bound**3 - lower_bound**3)

        # Calculate the weight = P(dL) / Q(dL)

        p_prob = 3 * new_distance**2 / (self.prior['distance'][1]**3
                                        - self.prior['distance'][0]**3)
        weight = p_prob / q_prob

        p_new = p.copy()
        p_new[self.param_idx['distance']] = new_distance

        # Rescale the waveform

        distance_scaling = np.float32(distance / new_distance)
        h_det_new = {}
        for ifo, h in h_det.items():
            h_det_new[ifo] = h * distance_scaling

        # weight = ((upper_bound**3 - lower_bound**3)
        #           / (self.prior['distance'][1]**3
        #              - self.prior['distance'][0]**3))

        return p_new, h_det_new, 1.0 # weight

    def calculate_threshold_standardizations(self, nsamples=100000):
        """Estimate the variances of the luminosity distance and the
        reduced basis coefficients, based on the SNR threshold. Also calculate
        the mean for the distance.

        This is needed in order to standardize inputs to the neural network,
        where the variance in each input is 1, and the mean is 0.

        Arguments:
            nsamples {int} -- number of waveforms to use in the estimate
                              (Default: 100000)
                              ('None' uses full training set)
        """

        # Generate detector waveforms, based on waveforms in the training set

        if nsamples is None or nsamples > len(self.train_selection):
            nsamples = len(self.train_selection)

        if self.domain == 'RB':
            waveform_size = self.Nrb
        elif self.domain == 'FD':
            waveform_size = self.Nf
        else:
            print('Not implemented.')
            return

        print('Calculating standardization variances based on SNR threshold'
              ' of {}'.format(self.snr_threshold))
        print('  Generating {} detector waveforms'.format(nsamples))

        # Create arrays
        h_detector = {}
        for ifo in self.detectors.keys():
            h_detector[ifo] = np.empty((nsamples, waveform_size),
                                       dtype=np.complex64)
        distances = np.empty(nsamples, dtype=np.float32)

        # Generate distances and waveforms
        for i in tqdm(range(nsamples)):
            p, h_det, _, _ = self.p_h_random_extrinsic(i, train=True)
            distances[i] = p[self.param_idx['distance']]
            for ifo, h in h_det.items():
                h_detector[ifo][i] = h

        print('  Calculating new standardization factors.')

        # Distance mean and standard deviation
        self.parameters_mean[self.param_idx['distance']] = np.mean(distances)
        self.parameters_std[self.param_idx['distance']] = np.std(distances)

        # Reduced basis standardization
        if self.domain == 'RB':
            for ifo, h_array in h_detector.items():
                self.basis.init_standardization(ifo, h_array, self._noise_std)


class WaveformDatasetTorch(Dataset):
    """Wrapper for a WaveformDataset to use with PyTorch DataLoader."""

    def __init__(self, wfd, train):

        self.wfd = wfd
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.wfd.train_selection)
        else:
            return len(self.wfd.test_selection)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.wfd.extrinsic_at_train:

            # Obtain parameters and waveform
            p, h, w, snr = self.wfd.p_h_random_extrinsic(idx, self.train)

            # Add noise, reshape, standardize
            x, y = self.wfd.x_y_from_p_h(p, h, add_noise=True)

            # Explicitly put the tensor w on the CPU, because default is CUDA.
            return (torch.from_numpy(y), torch.from_numpy(x),
                    torch.tensor(w, device='cpu'),
                    torch.tensor(snr, device='cpu'))

        else:
            # OLD CODE. REWORK FOR COMPATIBILITY.

            # Convert to index in wrapped WaveformDataset
            if self.train:
                wfd_idx = self.wfd.train_selection[idx]
            else:
                wfd_idx = self.wfd.test_selection[idx]

            params = (self.wfd.parameters[wfd_idx]
                      - self.wfd.parameters_mean) / self.wfd.parameters_std

            # Concatenate the waveforms from the different detectors
            wfs = []
            for d in self.wfd.detectors.keys():

                wf = self.wfd.h_dict[d][wfd_idx]/self.wfd._noise_std

                if self.wfd.domain == 'TD':
                    wfs.append(wf)

                elif self.wfd.domain == 'FD':

                    # Cut out the part of the waveforms below f_min
                    start_idx = int(self.wfd.f_min / self.wfd.delta_f)

                    wf_truncated = wf[start_idx:]
                    wfs.append(wf_truncated.real)
                    wfs.append(wf_truncated.imag)

            wf = np.concatenate(wfs, axis=-1)

            return (torch.from_numpy(wf), torch.from_numpy(params))
