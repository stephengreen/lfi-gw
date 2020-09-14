import numpy as np


def ptilde(d, dstar):
    """Unnormalized probability density.

    ptilde(d) = d^2 / (2 d / (3 dstar) + 1)^5

    Arguments:
        d {float} -- luminosity distance
        dstar {float} -- peak distance

    Returns:
        float -- unnormalized probability
    """
    return d**2 / (2*d/(3*dstar) + 1)**5


def normalization(dstar, dmin, dmax):
    """Normalization factor for BayesWave probability distribution.

    integral_dmin^dmax ptilde(d) dd

    Arguments:
        dstar {float} -- peak distance
        dmin {float} -- minimum distance
        dmax {float} -- maximum distance

    Returns:
        float -- normalization factor
    """
    return ((243*(dmax - dmin)*dstar**5 *
             (8*dmax**2*dmin**2*(dmax + dmin) +
              8*dmax*dmin*(dmax**2 + 7*dmax*dmin + dmin**2)*dstar +
              3*(dmax + dmin)*(dmax**2 + 16*dmax*dmin + dmin**2)*dstar**2 +
              18*(dmax**2 + dmax*dmin + dmin**2)*dstar**3)) /
            (2.*(2*dmax + 3*dstar)**4*(2*dmin + 3*dstar)**4))


def pdf(d, dstar, dmin, dmax):
    """Probability density function for BayesWave prior.

    Arguments:
        d {float} -- distance
        dstar {float} -- peak distance
        dmin {float} -- minimum distance
        dmax {float} -- maximum distance

    Returns:
        float -- probability
    """
    return ptilde(d, dstar) / normalization(dstar, dmin, dmax)


def cdf(d, dstar, dmin, dmax):
    """Cumulative distribution function for BayesWave prior.

    Arguments:
        d {float} -- distance
        dstar {float} -- peak distance
        dmin {float} -- minimum distance
        dmax {float} -- maximum distance

    Returns:
        float -- value of CDF
    """
    return normalization(dstar, dmin, d) / normalization(dstar, dmin, dmax)


# Functions needed to compute the inverse CDF. This requires solving a quartic
# equation. See notebook for details.


def alpha0(A):
    """Solution to the resolvent cubic.

    Arguments:
        A {float} --

    Returns:
        float -- real solution to the cubic
    """
    return (-(-2 + 4*np.sqrt(3)*np.sqrt(-A) +
              A**2*(((-1 + 2*np.sqrt(3)*np.sqrt(-A))*(1 + 12*A))/A**3)**(2/3) +
              ((1 + 12*A)*(-1 + 4*np.sqrt(3)*np.sqrt(-A) + 12*A)**2)**(1/3)) /
            (12.*(-1 + 2*np.sqrt(3)*np.sqrt(-A))*A))


def y0(A):
    """Solution to the quartic

    Arguments:
        A {float} --

    Returns:
        float -- relevant solution to the quartic
    """
    a0 = alpha0(A)
    return (- np.sqrt(a0) / np.sqrt(2) +
            (1/2) * np.sqrt(-1/A - 2*a0 - (2/(3*A))*np.sqrt(2/a0)))


def f(y):
    return -1/(2*y**2) + 2/(3*y**3) - 1/(4*y**4)


def A(u, dstar, dmin, dmax):
    ymin = 2*dmin / (3*dstar) + 1
    n = normalization(dstar, dmin, dmax)
    return f(ymin) + (2/(3*dstar))**3*n*u


def inverse_cdf(u, dstar, dmin, dmax):
    """Inverse CDF for Bayeswave prior

    Calculates CDF^{-1}(p)(u)

    Arguments:
        u {float} -- CDF(p)(d); 0 <= u <= 1
        dstar {float} -- peak distance
        dmin {float} -- minimum distance
        dmax {float} -- maximum distance

    Returns:
        float -- distance corresponding to u
    """
    if u < 0.0 or u > 1.0:
        raise ValueError('u should be between 0 and 1.')

    finv = y0(A(u, dstar, dmin, dmax))
    return 3*dstar/2 * (finv-1)
