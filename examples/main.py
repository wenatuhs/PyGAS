import os
import sys
import pickle
import warnings

from astracore import *
from plotplugins import *
from gaoptimizer import *

beamline = '/home/tangcx/projects/ttx/beamline'  # the beamline folder
simroot = '/home/tangcx/WORK/zhangzhe/data'  # the root simulation folder

# Constants
z_drift = 0.1  # m, end position relative to the linac exit
ntot = 50000  # number of macro particles
e_gun = 100  # MV/m, gun gradient
phi_l1 = 0  # deg, linac 1 phase
shift_lsol = 0.1  # m, position shift of the linac solenoid relative to the linac

# Variables
LT = [1e-3, 20e-3]  # ns, laser pulse length
SIGX = [0.1, 1.0]  # mm, laser radius (gaussian cut at 1 sigma)
PSOL = [0.21, 0.3]  # m, gun solenoid position
BSOL = [0.1, 0.3]  # T, gun solenoid strength
PHIGUN = [-20, 20]  # deg, gun launch phase
EL1 = [0, 35]  # MV/m, linac 1 gradient
PL1 = [1, 2]  # m, linac 1 position
BLSOL = [0, 0.3]  # T, linac solenoid strength


def recover(x, bound):
    return bound[0] + (bound[1] - bound[0]) * x


def gen_patch(x):
    lt = recover(x[0], LT)
    sig_x = recover(x[1], SIGX)
    pos_sol = recover(x[2], PSOL)
    b_sol = recover(x[3], BSOL)
    phi_gun = recover(x[4], PHIGUN)
    e_l1 = recover(x[5], EL1)
    pos_l1 = recover(x[6], PL1)
    b_lsol = recover(x[7], BLSOL)
    pos_lsol = pos_l1 + shift_lsol
    z_stop = 5.1

    patch = {'input': {'lt': lt,
                       'sig_x': sig_x,
                       'ipart': ntot},
             'newrun': {'auto_phase': True,
                        'zstop': z_stop},
             'cavity': {'lefield': True,
                        'maxe': [e_gun, e_l1],
                        'phi': [phi_gun, phi_l1],
                        'c_pos': [0, pos_l1]},
             'charge': {'lspch': True,
                        'lmirror': True},
             'solenoid': {'lbfield': True,
                          'maxb': [b_sol, b_lsol],
                          's_pos': [pos_sol, pos_lsol]}}

    return patch


def evaluate(x, sim):
    patch = gen_patch(x)
    try:
        core.run(patch, sim)
        data = core.get_data(sim, -1, 'g')
        emitx = nemit_t(data)[0]
        sig = sig_z(data)
        fitness = [emitx, sig]
        # Constrains
        if emitx > 2:  # emittance too large
            fitness[0] += (emitx - 2) ** 2
        if sig > 1.5:  # current too low
            fitness[1] += (sig - 1.5) ** 2
        elif sig < 0.2:  # current too high
            fitness[0] += (1 / sig - 5) ** 2
        if pnum(data) < 0.9 * ntot:  # particle loss
            fitness = [100, 100]  # almost death penalty
    except:
        fitness = [100, 100]  # almost death penalty
    return fitness


def parse_ga_input(arg1, arg2):
    try:
        npop = int(arg1)
    except:
        try:
            with open(arg1, 'rb') as f:
                _ngen = 0
                while True:
                    try:
                        npop = pickle.load(f)
                        _ngen += 1
                    except EOFError:
                        break
                    except:
                        raise
            if not _ngen:
                raise OptimizerError('ghist file {} is empty!'.format(arg1))
            print('Evolving based on the previous {0} generations from ghist file {1}...'.format(_ngen, arg1))
        except FileNotFoundError as exc:
            raise OptimizerError('error in reading ghist file {}!'.format(arg1)) from exc
    try:
        ngen = int(arg2)
    except ValueError as exc:
        raise OptimizerError('error in parsing number of generations!') from exc

    return npop, ngen


# Core initialization
core = AstraCore(beamline)

# Optimizer setup
opt = NSGAII(evaluate)
opt.NDIM = 8
opt.OBJ = (-1, -1)
opt.setup()

if __name__ == '__main__':
    # Parse the input arguments
    nargs = len(sys.argv[1:])
    if not nargs:
        raise OptimizerError('initial population (number) and number of generations are needed!')
    elif nargs == 1:
        fcount = 1
        while os.path.exists(HISTFNAME + (' {:d}'.format(fcount) if fcount > 1 else '')):
            fcount += 1
        fcount -= 1
        ghist = HISTFNAME + (' {:d}'.format(fcount) if fcount > 1 else '')
        npop, ngen = parse_ga_input(ghist, sys.argv[1])
    else:
        npop, ngen = parse_ga_input(sys.argv[1], sys.argv[2])
    if nargs > 2:
        _simroot = os.path.join(core.root, sys.argv[3])
        if os.path.exists(_simroot):
            simroot = _simroot
        else:
            warnings.warn('customized sim root folder not found, fallback to the default one!')

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
