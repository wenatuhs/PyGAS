import numpy as np
import scipy.constants as const

REST_ENERGY = const.m_e * const.c ** 2 / const.e * 1e-6  # 0.511 MeV


def freq2prd(f):
    """ Change frequency to time period.

    Keyword arguments:
    f -- frequency. [MHz]

    Returns:
    p -- time period. [ps]
    """
    p = 1e6 / f

    return p


def freq2lamb(f):
    """ Change frequency to wave length.

    Keyword arguments:
    f -- frequency. [MHz]

    Returns:
    lamb -- wave length. [mm]
    """
    lamb = 1e-3 * const.c / f

    return lamb


def lamb2en(lamb):
    """ Change wave length to energy.

    Keyword arguments:
    lamb -- wave length. [nm]

    Returns:
    energy -- energy. [eV]
    """
    energy = const.h * const.c / (lamb * 1e-9) / const.e

    return energy


def en2lamb(hv):
    """ Change photon energy to wavelength.

    Keyword arguments:
    hv -- photon energy. [eV]

    Returns:
    lamb -- wavelength. [nm]
    """
    lamb = const.h * const.c / (hv * 1e-9) / const.e

    return lamb


def time2len(t, beta):
    """ Change time length of the beam to spatial length.

    Keyword arguments:
    t -- time length. [ps]
    beta -- average relative velocity.

    Returns:
    z -- spatial length. [µm]
    """
    z = beta * const.c * t * 1e-6

    return z


def ek2gamma(ek):
    """ Convert Ekin to gamma.

    Keyword arguments:
    ek -- Ekin. [MeV]

    Returns:
    gamma -- relative energy.
    """
    return 1 + ek / REST_ENERGY


def beta2gamma(beta):
    """ Convert beta to gamma.

    Keyword arguments:
    beta -- relative velocity.

    Returns:
    gamma -- relative energy.
    """
    return 1 / np.sqrt(1 - beta ** 2)


def gamma2beta(gamma):
    """ Convert gamma to beta.

    Keyword arguments:
    gamma -- relative energy.

    Returns:
    beta -- relative velocity.
    """
    return np.sqrt(1 - 1 / gamma ** 2)
