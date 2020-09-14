import scipy
import numpy as np
import h5py
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


class SVDBasis(object):

    def __init__(self):
        self.whitening_dict = {}
        self.standardization_dict = {}
        self.T_matrices = None
        self.T_matrices_deriv = None

    def generate_basis(self, training_data, n, method='random'):
        """Generate the SVD basis from training data and store it.

        The SVD decomposition takes

        training_data = U @ diag(s) @ Vh

        where U and Vh are unitary.

        Arguments:
            training_data {array} -- waveforms in frequency domain

        Keyword Arguments:
            n {int} -- number of basis elements to keep.
                       n=0 keeps all basis elements. (default: {0})
        """

        if method == 'random':
            U, s, Vh = randomized_svd(training_data, n)

            self.Vh = Vh.astype(np.complex64)
            self.V = self.Vh.T.conj()

            self.n = n

        elif method == 'scipy':
            # Code below uses scipy's svd tool. Likely slower.

            U, s, Vh = scipy.linalg.svd(training_data, full_matrices=False)
            V = Vh.T.conj()

            if (n == 0) or (n > len(V)):
                self.V = V
                self.Vh = Vh
            else:
                self.V = V[:, :n]
                self.Vh = Vh[:n, :]

            self.n = len(self.Vh)

    def basis_coefficients_to_fseries(self, coefficients):
        """Convert from basis coefficients to frequency series.

        Arguments:
            coefficients {array} -- basis coefficients

        Returns:
            array -- frequency series
        """

        return coefficients @ self.Vh

    def fseries_to_basis_coefficients(self, fseries):
        """Convert from frequency series to basis coefficients.

        Arguments:
            fseries {array} -- frequency series

        Returns:
            array -- basis coefficients
        """

        return fseries @ self.V

    #
    # Time translation
    #

    def init_time_translation(self, t_min, t_max, Nt, f_grid):
        """Initialize the time translation matrices.

        The time translation in frequency domain corresponds to multiplication
        by e^{ - 2 pi i f dt }. If we only have waveforms in terms of basis
        coefficients, however, this is quite expensive: first one must
        transform to frequency domain, then time translate, then transform
        back to the reduced basis domain. Generally the dimensionality of
        FD waveforms will be much higher than the dimension of the reduced
        basis, so this is very costly.

        This function pre-computes N x N matrices in the reduced basis domain,
        where N is the dimension of the reduced basis. Matrices are computed
        at a discrete set of dt's. Later, interpolation is used to compute time
        translated coefficients away from these discrete points.

        Arguments:
            t_min {float} -- minimum value of dt
            t_max {float} -- maximum value of dt
            Nt {int} -- number of discrete points at which to compute matrices
            f_grid {array} -- frequencies at which FD waveforms are evaluated
        """

        self.t_grid = np.linspace(t_min, t_max, num=Nt, endpoint=True,
                                  dtype=np.float32)

        self.T_matrices = np.empty((Nt, self.n, self.n),
                                   dtype=np.complex64)
        self.T_matrices_deriv = np.empty((Nt, self.n, self.n),
                                         dtype=np.complex64)

        print('Building time translation matrices.')
        for i in tqdm(range(Nt)):

            # Translation by dt in FD is multiplication by e^{- 2 pi i f dt}
            T_fd = np.exp(- 2j * np.pi * self.t_grid[i] * f_grid)
            T_deriv_fd = - 2j * np.pi * f_grid * T_fd

            # Convert to FD, apply t translation, convert to reduced basis
            T_basis = (self.Vh * T_fd) @ self.V
            T_deriv_basis = (self.Vh * T_deriv_fd) @ self.V

            self.T_matrices[i] = T_basis
            self.T_matrices_deriv[i] = T_deriv_basis

    def time_translate(self, coefficients, dt, interpolation='linear'):
        """Calculate basis coefficients for a time-translated waveform.

        The new waveform h_new(t) = h_old(t - dt). In other words, if the
        original merger time is t=0, then the new merger time is t=dt.

        In frequency domain, this corresponds to multiplication by
        e^{ - 2 pi i f dt }.

        This method is capable of linear or cubic interpolation.

        Arguments:
            coefficients {array} -- basis coefficients of initial waveform
            dt {float} -- time translation

        Keyword Arguments:
            interpolation {str} -- 'linear' or 'cubic' interpolation
                                   (default: {'linear'})

        Returns:
            array -- basis coefficients of time-translated waveform
        """

        pos = np.searchsorted(self.t_grid, dt, side='right') - 1

        if self.t_grid[pos] == dt:

            # No interpolation needed
            translated = coefficients @ self.T_matrices[pos]

        else:
            t_left = self.t_grid[pos]
            t_right = self.t_grid[pos+1]

            # Interpolation parameter u(dt) defined so that:
            #           u(t_left) = 0
            #           u(t_right) = 1

            u = (dt - t_left) / (t_right - t_left)

            # Require coefficients evaluated on boundaries of interval
            y_left = coefficients @ self.T_matrices[pos]
            y_right = coefficients @ self.T_matrices[pos+1]

            if interpolation == 'linear':

                translated = y_left * (1 - u) + y_right * u

            elif interpolation == 'cubic':

                # Also require derivative of coefficients wrt dt
                dydt_left = coefficients @ self.T_matrices_deriv[pos]
                dydt_right = coefficients @ self.T_matrices_deriv[pos+1]

                # Cubic interpolation over interval
                # See https://en.wikipedia.org/wiki/Cubic_Hermite_spline

                h00 = 2*(u**3) - 3*(u**2) + 1
                h10 = u**3 - 2*(u**2) + u
                h01 = -2*(u**3) + 3*(u**2)
                h11 = u**3 - u**2

                translated = (y_left * h00
                              + dydt_left * h10 * (t_right - t_left)
                              + y_right * h01
                              + dydt_right * h11 * (t_right - t_left))

        return translated

    #
    # Whitening
    #
    # At present, we must know the fiducial and new noise PSD in advance, in
    # order to prepare the transformation matrices for reduced basis
    # coefficients. This is needed for dealing with detectors with different
    # PSDs.
    #
    # In the future, when we draw PSDs at random at train time, this will need
    # to be modified.
    #

    def init_whitening(self, ref_psd_name, ref_psd,
                       new_psd_name, new_psd):
        """Initialize whitening.

        Constructs and saves the whitening matrix for changing from a reference
        to a new noise PSD. This matrix acts on vectors of reduced basis
        coefficients.

        Arguments:
            ref_psd_name {str} -- label for fiducial PSD
            ref_psd {array} -- frequency series for fiducial PSd
            new_psd_name {str} -- label for new PSD
            new_psd {array} -- frequency series for new PSD
        """

        if ((new_psd_name != ref_psd_name)
                and (new_psd_name not in self.whitening_dict.keys())):

            # ref_psd = np.array(ref_psd)
            # new_psd = np.array(new_psd)

            whitening_FD = (ref_psd / new_psd) ** 0.5

            # Convert to float32 *after* dividing. PSDs can have very small
            # numbers.
            whitening_FD = whitening_FD.astype(np.float32)

            # Convert to RB representation
            whitening_RB = (self.Vh * whitening_FD) @ self.V

            whitening_RB = whitening_RB.astype(np.complex64)

            self.ref_psd_name = ref_psd_name
            self.whitening_dict[new_psd_name] = whitening_RB

    def whiten(self, coefficients, psd_name):
        """Whiten a waveform, given as a vector of reduced-basis coefficients.
        Waveform is assumed to already be white wrt reference PSD.

        Whitening must be first initialized with with init_whitening method.

        Arguments:
            coefficients {array} -- basis coefficients of initial waveform
            psd_name {str} -- label for new PSD

        Returns:
            array -- basis coefficients for whitened waveform
        """

        if psd_name != self.ref_psd_name:
            return coefficients @ self.whitening_dict[psd_name]

        else:
            return coefficients

    #
    # Truncation
    #

    def truncate(self, n):

        self.V = self.V[:, :n]
        self.Vh = self.Vh[:n, :]

        for ifo in self.standardization_dict.keys():
            self.standardization_dict[ifo] = self.standardization_dict[ifo][:n]

        for psd in self.whitening_dict.keys():
            self.whitening_dict[psd] = self.whitening_dict[psd][:n, :n]

        if self.T_matrices is not None:
            self.T_matrices = self.T_matrices[:, :n, :n]
            self.T_matrices_deriv = self.T_matrices_deriv[:, :n, :n]

        self.n = n

    #
    # Standardization
    #
    # Given a whitened noisy waveform, we want to rescale each component to
    # have unit variance. This is to improve neural network training. The mean
    # should already be zero.
    #

    def init_standardization(self, ifo, h_array, noise_std):

        # Standard deviation of data. Divide by sqrt(2) because we want real
        # and imaginary parts to have unit standard deviation.
        std = np.std(h_array, axis=0) / np.sqrt(2)

        # Total standard deviation
        std_total = np.sqrt(std**2 + noise_std**2)

        self.standardization_dict[ifo] = 1.0 / std_total

    def standardize(self, h, ifo):

        return h * self.standardization_dict[ifo]

    #
    # File I/O
    #

    def save(self, directory='.', filename='reduced_basis.hdf5'):

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        f = h5py.File(p / filename, 'w')

        f.create_dataset('V', data=self.V,
                         compression='gzip', compression_opts=9)

        if self.standardization_dict != {}:
            std_group = f.create_group('std')
            for ifo, std in self.standardization_dict.items():
                std_group.create_dataset(ifo, data=std,
                                         compression='gzip',
                                         compression_opts=9)

        f.close()

    def load(self, directory='.', filename='reduced_basis.hdf5'):

        p = Path(directory)

        f = h5py.File(p / filename, 'r')
        self.V = f['V'][:, :]

        if 'std' in f.keys():
            std_group = f['std']
            for ifo in std_group.keys():
                self.standardization_dict[ifo] = std_group[ifo][:]

        f.close()

        self.Vh = self.V.T.conj()
        self.n = len(self.Vh)
