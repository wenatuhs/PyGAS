import os
import sys
import pickle
import warnings

from astracore import *
from plotplugins import *
from gaoptimizer import *

beamline = '/home/tangcx/projects/twofreqrfgun/beamline'  # the beamline folder
simroot = '/home/tangcx/WORK/data'  # the root simulation folder

# Constants
eta = 0.48  # sigma ratio of the truncated gaussian dist
norm_emit = 0.9  # mm.mrad/mm, normalized emittance
z_stop = 11.0  # m, end position
ntot = 3000  # number of macro particles
phi_l1 = 0  # deg, linac 1 phase
phi_l2 = 0  # deg, linac 2 phase

# Variables
LT = [1e-3, 20e-3]  # ns, laser pulse length
SIGX = [0.1, 1.0]  # mm, laser radius (gaussian cut at 1 sigma)
PSOL = [0.2, 0.3]  # m, gun solenoid position
BSOL = [0.15, 0.25]  # T, gun solenoid strength
PHIGUN = [-10, 35]  # deg, gun launch phase
PHICAV = [-180, 0]  # deg, high order cavity phase
EGUN = [80, 130]  # MV/m, gun gradient
ECAV = [0, 200]  # MV/m, high order cavity gradient
EL1 = [10, 50]  # MV/m, linac 1 gradient
EL2 = [10, 50]  # MV/m, linac 2 gradient
PL1 = [1.5, 2.5]  # m, linac 1 position
PL2 = [5.5, 7.0]  # m, linac 2 position


def recover(x, bound):
    return bound[0] + (bound[1] - bound[0]) * x


def gen_patch(x):
    lt = recover(x[0], LT)
    sig_x = recover(x[1], SIGX)
    pos_sol = recover(x[2], PSOL)
    b_sol = recover(x[3], BSOL)
    phi_gun = recover(x[4], PHIGUN)
    phi_cav = recover(x[5], PHICAV)
    e_gun = recover(x[6], EGUN)
    e_cav = recover(x[7], ECAV)
    e_l1 = recover(x[8], EL1)
    e_l2 = recover(x[9], EL2)
    pos_l1 = recover(x[10], PL1)
    pos_l2 = recover(x[11], PL2)

    patch = {'input': {'lt': lt,
                       'sig_x': sig_x,
                       'nemit_x': sig_x * eta * norm_emit,
                       'ipart': ntot},
             'newrun': {'auto_phase': True,
                        'zstop': z_stop},
             'cavity': {'lefield': True,
                        'maxe': [e_gun, e_cav, e_l1, e_l2],
                        'phi': [phi_gun, phi_cav, phi_l1, phi_l2],
                        'c_pos': [0, 0, pos_l1, pos_l2]},
             'charge': {'lspch': True,
                        'lmirror': True},
             'solenoid': {'lbfield': True,
                          'maxb': [b_sol],
                          's_pos': [pos_sol]}}

    return patch


def evaluate(x, sim):
    patch = gen_patch(x)
    try:
        core.run(patch, sim)
        data = core.get_data(sim, -1, 'g')
        fitness = [nemit_t(data)[0], current_r(data), skewness(data)]
        if pnum(data) < 0.9 * ntot:  # a small penalty!
            fitness[0] += 1
            fitness[1] -= 100
            fitness[2] += 1
    except:
        fitness = [100, 0, 0]  # almost death penalty
    return fitness


def parse_ga_input(arg1, arg2):
    try:
        npop = int(arg1)
    except:
        try:
            with open(arg1, 'rb') as f:
                _ngen, npop = pickle.load(f)
            print('Evolving based on the previous {} generations...'.format(_ngen))
        except FileNotFoundError as exc:
            raise OptimizerError('error in reading population file {}!'.format(arg1)) from exc
        except:
            raise
    try:
        ngen = int(arg2)
    except ValueError as exc:
        raise OptimizerError('error in parsing number of generations!') from exc
    except:
        raise

    return npop, ngen


# Core initialization
core = AstraCore(beamline)

# Optimizer setup
opt = NSGAII(evaluate)
opt.NDIM = 12
opt.OBJ = (-1, 1, -1)
opt.setup()

if __name__ == '__main__':
    # Parse the input arguments
    nargs = len(sys.argv[1:])
    if not nargs:
        raise OptimizerError('initial population (number) and number of generations are needed!')
    elif nargs == 1:
        npop, ngen = parse_ga_input('pop', sys.argv[1])
    else:
        npop, ngen = parse_ga_input(sys.argv[1], sys.argv[2])
    if nargs > 2 and os.path.exists(sys.argv[3]):
        simroot = sys.argv[3]

    # Let's rock!
    opt.evolve(npop, ngen, pre=simroot)

    # Record
    with open('gapop', 'w') as f:
        f.write(str(opt.pop))
    with open('galog', 'w') as f:
        f.write(str(opt.log))
    with open('gafit', 'w') as f:
        fit = [ind.fitness.values for ind in opt.pop]
        f.write(str(fit))
