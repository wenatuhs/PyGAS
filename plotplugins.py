from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from statplugins import *
from astracore import PostProcessWarning


def _cal_color(xy, norm=False):
    """ Calculate colors based on the density of the 2D data points.

    Keyword arguments:
    xy -- 2D array, 2*n.
    norm -- [False] if normalize the xy data.

    Returns:
    color -- 1D array, 1*n.
    """
    color = gaussian_kde(xy)(xy)
    if norm:
        cmin = np.min(color)
        cmax = np.max(color)
        color = (color - cmin) / (cmax - cmin)

    return color


def _norm_colors(colors):
    """ Normalize the colors in the color list.

    Keyword arguments:
    colors -- a list of color.

    Returns:
    colors -- a list of normalized color.
    """
    minmax = np.array([[np.min(color), np.max(color)] for color in colors])
    cmin = np.min(minmax[:, 0])
    cmax = np.max(minmax[:, 1])

    def norm(x):
        return (x - cmin) / (cmax - cmin)

    colors = [norm(color) for color in colors]

    return colors


def _gen_ellipse(twiss, ep=1, num=100):
    """ Generate the phase ellipse data points based on the twiss paras and the emittance.

    Keyword arguments:
    twiss -- twiss paras.
    ep -- [1] emittance.
    num -- [100] how many points to generate.

    Returns:
    elli -- 2*n data points array.
    """
    a, b, c = twiss

    t = np.linspace(0, 2 * np.pi, num)
    t0 = np.arctan(a)
    x = np.sqrt(b * ep) * np.cos(t)
    y = np.sqrt(c * ep) * np.sin(t - t0)

    return np.vstack([x, y])


def phase_space_t(data, kind='n', trace=True, save=False):
    """ Visualize the transverse phase-space.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.
    kind -- ['n'] normalized phase-space or geometry phase-space.
        'n' or 'g'
    save -- save the figure of not.
    """
    # Prepare the data and texts
    x, y, z, px, py, pz, t = data

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    if kind == 'g':
        xp, yp = px / pz * 1e3, py / pz * 1e3
        xlabs = ['x (mm)', 'y (mm)']
        ylabs = ['px/pz (mrad)', 'py/pz (mrad)']
        labs = ['geo phase space x', 'geo phase space y']
        if trace:
            twiss = twiss_paras(data)
            emits = gemit_t(data)
    else:
        if kind != 'n':
            warnings.warn('phase space type {} is not supported,'
                          'fallback to normalized phase space!'.format(kind),
                          PostProcessWarning)

        xp, yp = px / REST_ENERGY * 1e3, py / REST_ENERGY * 1e3
        xlabs = ['x (mm)', 'y (mm)']
        ylabs = ['px/mc (mrad)', 'py/mc (mrad)']
        labs = ['norm phase space x', 'norm phase space y']
        if trace:
            twiss = ntwiss_paras(data)
            emits = nemit_t(data)

    colors = [_cal_color(np.array([x, xp])), _cal_color(np.array([y, yp]))]
    colors = _norm_colors(colors)
    xxpyyp = [[x, xp], [y, yp]]
    cmaps = ['Blues', 'Greens']

    # Begin to plot
    for ax, xlab, ylab, xxp, color, lab, cm in zip(axs, xlabs, ylabs, xxpyyp,
                                                   colors, labs, cmaps):
        ax.set(xlabel=xlab, ylabel=ylab)
        ax.scatter(xxp[0], xxp[1], c=color, marker='+', label=lab,
                   vmin=0, vmax=1, cmap=cm)
    if trace:
        ori = np.mean(xxpyyp, 2)
        for ax, tw, ep, o in zip(axs, twiss, emits, ori):
            _x, _y = _gen_ellipse(tw, ep)
            ax.plot(2 * _x + o[0], 2 * _y + o[1], c='dimgrey', lw=0.5)

    clegs = ['b', 'g']
    for ax, cleg in zip(axs, clegs):
        ax.axhline(0, color='k', ls=':', lw=0.5)
        ax.axvline(0, color='k', ls=':', lw=0.5)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1 - x0) / (y1 - y0))

        leg = ax.legend(loc=0)
        leg.legendHandles[0].set_color(cleg)

    fig.tight_layout()

    if save:
        fig.savefig('phase_space_t.pdf', bbox_inches='tight')
    plt.show()


def phase_space_l(data, save=False):
    """ Visualize the longitudinal phase-space.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.
    save -- save the figure of not.
    """
    x, y, z, px, py, pz, t = data

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    dz = z - np.mean(z)
    dp = pz / np.mean(pz) - 1
    colors = _cal_color(np.array([dz, dp]), True)

    ax.set(xlabel='dz (mm)', ylabel='dpz/pz')
    ax.scatter(dz, dp, c=colors, marker='+', label='phase space z',
               vmin=0, vmax=1, cmap='Reds')

    ax.axhline(0, color='k', ls=':', lw=0.5)
    ax.axvline(0, color='k', ls=':', lw=0.5)

    leg = ax.legend(loc=0)
    leg.legendHandles[0].set_color('r')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))

    fig.tight_layout()

    if save:
        fig.savefig('phase_space_l.pdf', bbox_inches='tight')
    plt.show()


def dist_l(data, num=20, save=False):
    """ Plot the longitudinal charge density distribution.

    Keyword arguments:
    data -- x, y, z, px, py, pz, t.
        x, y, z -- mm.
        px, py, pz -- MeV/c.
        t -- ps.
    num -- [20] how many bins to use in hist.
    save -- [False] save the plot.
    """
    x, y, z, px, py, pz, t = data

    fig, ax = plt.subplots(1, 1)
    ax.set(xlabel='dz (mm)', ylabel='count')

    count, bins = np.histogram(z - np.mean(z), num)
    bins = (bins[:-1] + bins[1:]) / 2
    skew = skewness(data, num)
    ax.plot(bins, count, 'b-', lw=1,
            marker='o', mec='none', mfc=[1, 0, 0, 0.5])
    ax.fill_between(bins, count, 0, alpha=0.1, edgecolor='none',
                    label='longitudinal dist\nskewness = {0:.3f}'
                          '\ncurrent = {1:.1f}'.format(skew, current_r(data)))
    ax.axvline(0, color='k', ls=':', lw=0.5)

    ax.legend(loc=0)
    fig.tight_layout()

    if save:
        fig.savefig('dist_l.pdf', bbox_inches='tight')
    plt.show()
