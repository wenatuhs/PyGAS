from astracore import *
from plotplugins import *
from gaoptimizer import *

root = '/Users/wena/Desktop/S:X-GUN/sims/astra'
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


def evaluate(x):
    sim = 'test2'
    patch = gen_patch(x)
    core.run(patch, sim)
    data = core.get_data(sim, -1, 'g')
    return [nemit_t(data)[0], current_r(data), skewness(data)]


if __name__ == '__main__':
    # Optimizer setup
    opt = NSGAII(evaluate)
    opt.NDIM = 8
    opt.OBJ = (-1, 1, -1)
    opt.setup()

    # Let’s rock!
    opt.evolve(12, 2)

    # Post-process
    xyz = np.array([ind.fitness.values for ind in opt.pop])
    plt.plot(xyz[:, 0], xyz[:, 1], 'b.')
    print(xyz)
    core.run(gen_patch(opt.pop[0]), 'cool2', 1)
    data = core.get_data('cool2', -1, 'g')
    dist_l(data, 20)
