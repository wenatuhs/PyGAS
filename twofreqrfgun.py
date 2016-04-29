import sys
import warnings

from astracore import *
from plotplugins import *
from gaoptimizer import *

root = '/home/tangcx/projects/twofreqrfgun/beamline'
core = AstraCore(root)

# Constants
eta = 0.48  # sigma ratio of the truncated gaussian dist
norm_emit = 0.9  # mm.mrad/mm
e_l1 = 0  # MV/m
e_l2 = 0  # MV/m
pos_l2 = 5.5  # m, linac 2 position
z_stop = 1.9  # m, end position
pos_l1 = 1.9  # m, linac 1 position
phi_l1 = 0  # deg, linac 1 phase
phi_l2 = 0  # deg, linac 2 phase

# Variables
LT = [1e-3, 20e-3]  # ps, laser pulse length
SIGX = [0.1, 1.0]  # mm, laser radius (gaussian cut at 1 sigma)
PHIGUN = [15, 60]  # deg, gun launch phase
PHICAV = [0, 180]  # deg, high order cavity phase
EGUN = [80, 130]  # MV/m
ECAV = [0, 100]  # MV/m
PSOL = [0.2, 0.3]  # m, gun solenoid position
BSOL = [0.15, 0.25]  # T, gun solenoid strength


def recover(x, bound):
    return bound[0] + (bound[1] - bound[0]) * x


def gen_patch(x):
    lt = recover(x[0], LT)
    sig_x = recover(x[1], SIGX)
    phi_gun = recover(x[2], PHIGUN)
    phi_cav = recover(x[3], PHICAV)
    e_gun = recover(x[4], EGUN)
    e_cav = recover(x[5], ECAV)
    pos_sol = recover(x[6], PSOL)
    b_sol = recover(x[7], BSOL)

    patch = {'input': {'lt': lt,
                       'sig_x': sig_x,
                       'nemit_x': sig_x * eta * norm_emit},
             'newrun': {'phase_scan': False,
                        'auto_phase': False,
                        'zstop': z_stop},
             'cavity': {'maxe': [e_gun, e_cav, e_l1, e_l2],
                        'phi': [phi_gun, phi_cav, phi_l1, phi_l2],
                        'c_pos': [0, 0, pos_l1, pos_l2]},
             'solenoid': {'lbfield': True,
                          'maxb': [b_sol],
                          's_pos': [pos_sol]}}

    return patch


def evaluate(x, sim):
    patch = gen_patch(x)
    core.run(patch, sim)
    data = core.get_data(sim, -1, 'g')
    return [nemit_t(data)[0], current_r(data), skewness(data)]


# Optimizer setup
opt = NSGAII(evaluate)
opt.NDIM = 8
opt.OBJ = (-1, 1, -1)
opt.setup()

if __name__ == '__main__':
    # Let’s rock!
    try:
        npop, ngen = (int(n) for n in sys.argv[1:])
    except:
        npop, ngen = 24, 2  # for test
        warnings.warn("fallback to test case with npop={0}, ngen={1}!".format(npop, ngen), OptimizerWarning)
    opt.evolve(npop, ngen, pre='/home/tangcx/WORK/data')

    # Record
    with open('gapop', 'w') as f:
        f.write(str(opt.pop))
    with open('galog', 'w') as f:
        f.write(str(opt.log))
    with open('gafit', 'w') as f:
        fit = [ind.fitness.values for ind in opt.pop]
        f.write(str(fit))
