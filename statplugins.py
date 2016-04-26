import numpy.fft as fft
from scipy.interpolate import interp1d
import warnings
from functools import wraps

from astracore import AstraCoreWarning, PostProcessError, PostProcessWarning
from physicshelper import *


def _digitize(x, num):
    """ My digitize version to categorize data.

    Keyword arguments:
    x -- the data to be digitized.
    num -- the number of the slices.

    Returns:
    d -- the digitized data.
    """
    bins = np.linspace(np.min(x), np.max(x), num + 1)
    d = np.digitize(x, bins)
    d[d == num + 1] = num  # the last range should be closed

    return d


def _cal_sig_nemit(x, px):
    """ Calculate sigma and normalized emittance.

    Keyword arguments:
    x -- x array in mm.
    px -- px array in MeV/c

    Returns:
    sig, nemit -- sigma in mm and normalized emittance in mm.mrad.
    """
    m = np.vstack((x, px))
    cov_m = np.cov(m, bias=1)

    sig = np.sqrt(cov_m[0, 0])  # [mm]
    nemit = np.sqrt(np.linalg.det(cov_m)) * 1e3 / REST_ENERGY  # [mm.mrad]

    return np.array([sig, nemit])


def _cal_sig(x):
    """ Calculate transverse sigma.

    Keyword arguments:
    x -- x array in mm.

    Returns:
    sig -- sigma in mm.
    """
    sig = np.std(x)

    return sig


def _cal_pos(x):
    """ Calculate transverse position.

    Keyword arguments:
    x -- x array in mm.

    Returns:
    pos -- pos in mm.
    """
    pos = np.mean(x)

    return pos


def _cal_delta(px):
    """ Calculate sigma and normalized emittance.

    Keyword arguments:
    px -- px array in MeV/c.

    Returns:
    delta -- delta in mrad.
    """
    delta = np.std(px) * 1e3 / REST_ENERGY

    return delta


def _cal_cross(x, px):
    """ Calculate the cross item.

    Keyword arguments:
    x -- x array in mm.
    px -- px array in MeV/c.

    Returns:
    xpx -- cross item in mm.mrad.
    """
    m = np.vstack((x, px))
    cov_m = np.cov(m, bias=1)

    xpx = cov_m[0, 1] * 1e3 / REST_ENERGY

    return xpx


def _cal_ek(px, py, pz):
    """ Calculate the kinetic energy from the particle momentum coordinates.

    Keyword arguments:
    px, py, pz -- MeV/c.

    Returns:
    Ek -- kinetic energy in MeV.
    """
    ek = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + REST_ENERGY ** 2) - REST_ENERGY  # [MeV]

    return ek


def _cal_emit(x, xp):
    """ Calculate the emittance.

    Keyword arguments:
    x -- x array in mm.
    xp -- xp array in mrad.

    Returns:
    emit -- emittance in mm.mrad.
    """
    m = np.vstack((x, xp))
    cov_m = np.cov(m, bias=1)

    emit = np.sqrt(np.linalg.det(cov_m))  # [mm.mrad]

    return emit


def _cal_twiss(x, xp):
    """ Calculate the Twiss parameters.

    Keyword arguments:
    x -- x array in mm.
    xp -- xp array in mrad.

    Returns:
    twiss_paras -- alpha, beta, gamma. [1, m, 1/m]
    """
    m = np.vstack((x, xp))
    cov_m = np.cov(m, bias=1)

    emit = np.sqrt(np.linalg.det(cov_m))  # [mm.mrad]
    beta = cov_m[0, 0] / emit  # m
    gamma = cov_m[1, 1] / emit  # 1/m
    alpha = -cov_m[0, 1] / emit  # 1

    return np.array([alpha, beta, gamma])


def _cal_ftime(z, px, py, pz):
    """ Calculate the particle flight time relative to the reference particle.

    Keyword arguments:
    z -- z array in mm.
    px, py, pz -- MeV/c.

    Returns:
    ft -- flight time in ps.
    """
    rp = const.m_e * const.c / const.e * 1e-6  # 0.511/c
    _z = z - np.mean(z)
    _vz = pz / rp / ek2gamma(_cal_ek(px, py, pz))
    ft = _z / _vz * 1e9  # [ps]

    return ft


def _cal_skew(z, method='fourier'):
    """ Calculate the skewness of the given 1D distribution.
    Smaller skewness means the distribution to be more symmetrical.

    Keyword arguments:
    z -- the 1D distribution to be examined.
    method -- ['fourier'] the test method to be applied.
        'fourier', 'evenodd'

    Returns:
    skew -- skewness.
    """
    if method == 'fourier':
        spec = fft.fft(z)
        e_imag = np.sum(spec.imag ** 2)
        e_real = np.sum(spec.real ** 2)

        skew = np.log10(1 / (1 + e_real / e_imag))
    elif method == 'evenodd':  # this one is proved to be not accurate enough
        z_odd = (z - z[::-1]) / 2
        z_even = (z + z[::-1]) / 2
        e_odd = np.sum(z_odd ** 2)
        e_even = np.sum(z_even ** 2)

        skew = np.log10(1 / (1 + e_even / e_odd))
    else:
        msg = 'methode {} is not supported yet, fallback to fourier method!'.format(method)
        warnings.warn(msg, AstraCoreWarning)

        skew = _cal_skew(z)

    return skew


def _cal_elli_val(x, xp):
    """ Calculate the 'ellipse value' for each data point in the phase space.

    Keyword arguments:
    x -- x array in mm.
    xp -- xp array in mrad.

    Returns:
    ep -- ellipse values.
    """
    a, b, c = _cal_twiss(x, xp)
    _x, _xp = x - np.mean(x), xp - np.mean(xp)

    return c * _x ** 2 + 2 * a * _x * _xp + b * _xp ** 2


def _empty_zero_mean(a):
    """ A temporal version of mean(), to redefine mean of empty list handling.

    Keyword arguments:
    a -- 1D array.

    Returns:
    m -- mean of the given 1D array or list, if the 1D array is empty, returns 0.
    """
    if len(a):
        return np.mean(a)
    else:
        return 0


def _interp_emit(x, xp, num=100):
    """ Get the interp func for the ratio emittance.

    Keyword arguments:
    x -- x array in mm.
    xp -- xp array in mrad.
    num -- [100] how many points to use in the interpolation.

    Returns:
    interp_func -- the interp function.
    """
    evs = _cal_elli_val(x, xp)
    evmax = np.max(evs)
    ratio = np.linspace(0, 1, num)

    ev_list = [evs[evs <= r * evmax] for r in ratio]

    count_emits = np.array([[len(ev), _empty_zero_mean(ev) / 2] for ev in ev_list]).transpose()
    tot_count = count_emits[0, -1]

    return interp1d(count_emits[0] / tot_count, count_emits[1])


def check(allow_empty=True):
    def check_func(func):
        """ Decorator that checks the data before pass it to func.

        Keyword arguments:
        func -- the given function.

        Returns:
        check_func -- func with data validation check.
        """

        @wraps(func)
        def func_valid(*args, **kwargs):
            """ Check the validity of the given data.
            """
            try:
                if 'data' in kwargs.keys():
                    data = kwargs['data']
                else:
                    data = args[0]
            except IndexError:
                warnings.warn("data not found, ignoring the validation check!",
                              PostProcessWarning)

            if data.size:
                r, c = data.shape
                if r != 7:
                    raise PostProcessError(
                        'data should contain x, y, z, px, py, pz, t, please check!')
            elif allow_empty:
                warnings.warn("data is empty, the post-process could have troubles!",
                              PostProcessWarning)
            else:
                raise PostProcessError(
                    'data is not allowed to be empty!')
            return func(*args, **kwargs)

        return func_valid

    if type(allow_empty) is not bool:
        func, allow_empty = allow_empty, True
        return check_func(func)
    else:
        return check_func


@check
def slice_data(data, num=1, dim='z'):
    """ Slice the data into a given number slices based on the given dimension.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.
    num -- [1] number of slices.
    dim -- ['z'] dimension on which to slice the data.
        'x', 'y', 'z', 'px', 'py', 'pz', 't', 'E' (here E is Ek)

    Returns:
    data_slice -- sliced data list.
    """
    x, y, z, px, py, pz, t = data

    if num == 1:
        data_sliced = [data]
    elif data.size:
        if dim == 'z':
            proj = z
        elif dim == 't':
            proj = t
        elif dim == 'E':
            proj = _cal_ek(px, py, pz)
        elif dim == 'I':
            proj = _cal_ftime(z, px, py, pz)  # [ps]
        else:
            msg = 'dimension type is not supported, fallback to z-slice!'
            warnings.warn(msg, AstraCoreWarning)
            proj = z
        d = _digitize(proj, num)
        data_sliced = [data[:, d == i + 1] for i in range(num)]
    else:
        raise PostProcessError(
            'data is not allowed to be empty!')

    return data_sliced


def sliced(stat):
    """ Decorator that returns sliced version of the given function stat.

    Keyword arguments:
    stat -- the given statistic function.

    Returns:
    sliced_stat -- the sliced version of function stat.
    """

    @wraps(stat)
    def sliced_stat(data, num=1, dim='z'):
        """ Sliced statistics of the given data.

        Keyword arguments:
        data -- the given data.
        num -- [1] the slice number.
        dim -- ['z'] the dimension.

        Returns:
        sliced_stat_data -- an array of sliced data statistics.
        """
        slices = slice_data(data, num, dim)

        if len(slices) == 1:
            return np.array([stat(s) for s in slices])[0]
        else:
            return np.array([stat(s) for s in slices])

    return sliced_stat


@sliced
@check
def avg_ek(data):
    """ Calculate the average kinetic energy from the particle coordinates.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    Ek -- kinetic energy in MeV.
    """
    x, y, z, px, py, pz, t = data

    return np.mean(_cal_ek(px, py, pz))


@sliced
@check
def sig_ek(data):
    """ Calculate the energy spread from the particle coordinates.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    sigma_Ek -- energy spread in MeV.
    """
    x, y, z, px, py, pz, t = data

    return np.std(_cal_ek(px, py, pz))


@sliced
@check
def avg_gamma(data):
    """ Calculate the average relative energy.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    gamma -- relative energy.
    """
    return ek2gamma(avg_ek(data))


@sliced
@check
def avg_beta(data):
    """ Calculate the average relative velocity.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    beta -- relative velocity.
    """
    gamma = avg_gamma(data)

    return gamma2beta(gamma)


@sliced
@check
def avg_beta_gamma(data):
    """ Calculate the average beta*gamma.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    beta*gamma -- relative velocity times relative energy.
    """
    gamma = avg_gamma(data)
    beta = gamma2beta(gamma)

    return beta * gamma


@sliced
@check
def sig_t(data):
    """ Calculate sigma in transverse direction.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    sig -- sigma in mm.
    """
    x, y, z, px, py, pz, t = data

    sig = np.array([_cal_sig(x), _cal_sig(y)])

    return sig


@sliced
@check
def pos_t(data):
    """ Calculate position in transverse direction.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    pos -- pos in mm.
    """
    x, y, z, px, py, pz, t = data

    pos = np.array([_cal_pos(x), _cal_pos(y)])

    return pos


@sliced
@check
def delta_t(data):
    """ Calculate momentum spread in transverse direction.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    pos -- pos in mm.
    """
    x, y, z, px, py, pz, t = data

    delta = np.array([_cal_delta(px), _cal_delta(py)])

    return delta


@sliced
@check
def nemit_t(data):
    """ Calculate normalized emittance in transverse direction.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    nemits -- emittance array for x and y direction..
    """
    x, y, z, px, py, pz, t = data

    xp, yp = px / REST_ENERGY * 1e3, py / REST_ENERGY * 1e3

    return np.array([_cal_emit(x, xp), _cal_emit(y, yp)])


@sliced
@check
def x_gemit_t(data):
    """ Calculate geometry emittance in transverse direction.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    gemits -- emittance array for x and y direction..
    """
    return nemit_t(data) / avg_beta_gamma(data)


@sliced
@check
def gemit_t(data):
    """ Calculate geometry emittance in transverse direction.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    gemits -- emittance array for x and y direction..
    """
    x, y, z, px, py, pz, t = data

    xp, yp = px / pz * 1e3, py / pz * 1e3

    return np.array([_cal_emit(x, xp), _cal_emit(y, yp)])


@sliced
@check
def ntwiss_paras(data):
    """ Calculate the normalized Twiss parameters.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    ntwiss -- array of normalized alpha, beta, gamma for x and y direction.
    """
    x, y, z, px, py, pz, t = data

    xp, yp = px / REST_ENERGY * 1e3, py / REST_ENERGY * 1e3

    ntwiss = np.array([_cal_twiss(x, xp), _cal_twiss(y, yp)])

    return ntwiss


@sliced
@check
def x_twiss_paras(data):
    """ Calculate the Twiss parameters.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    twiss -- array of alpha, beta, gamma for x and y direction.
    """
    twiss = ntwiss_paras(data)
    bg = avg_beta_gamma(data)

    twiss[:, 1] *= bg
    twiss[:, 2] /= bg
    #     x, y, z, px, py, pz, t = data

    #     twiss = np.array([_cal_twiss(x, px, pz), _cal_twiss(y, py, pz)])

    return twiss


@sliced
@check
def twiss_paras(data):
    """ Calculate the Twiss parameters.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    twiss -- array of alpha, beta, gamma for x and y direction.
    """
    x, y, z, px, py, pz, t = data

    xp, yp = px / pz * 1e3, py / pz * 1e3

    gtwiss = np.array([_cal_twiss(x, xp), _cal_twiss(y, yp)])

    return gtwiss


@sliced
@check
def pos_z(data):
    """ Calculate average z position.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    pos_z -- z in mm.
    """
    x, y, z, px, py, pz, t = data

    return np.mean(z)


@sliced
@check
def sig_z(data):
    """ Calculate bunch length.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    sig_z -- z in mm.
    """
    x, y, z, px, py, pz, t = data

    return np.std(z)


@sliced
@check
def pnum(data):
    """ Calculate particle number.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    num -- number of particles.
    """

    return data.shape[1]


@check
def current_r(data):
    """ Calculate relative current.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.

    Returns:
    sig_z -- z in mm.
    """
    x, y, z, px, py, pz, t = data

    q = pnum(data)
    ft = _cal_ftime(z, px, py, pz)
    dis = np.max(ft) - np.min(ft)

    return q / dis


@check
def skewness(data, num=20, method='fourier'):
    """ Calculate the skewness of the longitudinal distribution.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.
    num -- [20] how many bins to use in histogram.
    method -- ['fourier'] the test method to be applied.
        'fourier', 'evenodd'

    Returns:
    skew -- skewness.
    """
    z = pnum(data, num)

    return _cal_skew(z, method)


@check
def func_ratio_emit_t(data, kind='n', num=100):
    """ Get interp functions that calculate the ratio emittance.
    Say, 90% emittance, etc.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.
    kind -- ['n'] emittance type.
        'n': normalized emittance.
        'g': geometry emittance.
    num -- [100] how many points to use in the interpolation.

    Returns:
    interp_x, interp_y -- the interp functions.
    """
    x, y, z, px, py, pz, t = data

    if kind == 'g':
        xp, yp = px / pz * 1e3, py / pz * 1e3
    else:
        if kind != 'n':
            warnings.warn('emittance type {} is not supported,' \
                          'fallback to normalized emittance!'.format(kind),
                          PostProcessWarning)
        xp, yp = px / REST_ENERGY * 1e3, py / REST_ENERGY * 1e3

    return _interp_emit(x, xp, num), _interp_emit(y, yp, num)
