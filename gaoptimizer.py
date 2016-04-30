import os
import pickle
import random
import warnings

import numpy as np
from deap import base
from deap import creator
from deap import tools
from scoop import futures
from tqdm import tqdm


class OptimizerError(Exception):
    """ Optimizer error class.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class OptimizerWarning(UserWarning):
    """ Optimizer warning class.
    """
    pass


class NSGAII:
    """ NSGA-II optimizer.
    """

    def __init__(self, evaluate=None):
        """ Initialize the optimizer.

        Keyword arguments:
        evaluate -- [None] the evaluate function.
        """
        self.NDIM = 30
        self.OBJ = (-1.0, -1.0)
        self.ETAC = 10
        self.ETAM = 10
        self.CXPB = 0.9
        self.MPB = 0.3
        self.MIDPB = 0.5
        self.evaluate = evaluate
        self.toolbox = None
        self.pop = None
        self.log = None
        self.setup()

    def setup(self):
        """ Call this method when any member variable changes to update
        the NSGA-II arguments.
        """
        try:
            del creator.Quality
        except AttributeError:
            pass
        try:
            del creator.Individual
        except AttributeError:
            pass
        creator.create("Quality", base.Fitness, weights=self.OBJ)
        creator.create("Individual", list, fitness=creator.Quality)

        toolbox = base.Toolbox()

        toolbox.register("norm_var", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.norm_var, self.NDIM)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # toolbox.register("evaluate", benchmarks.zdt2)
        if not self.evaluate:
            warnings.warn("evaluate function not defined!", OptimizerWarning)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                         low=0, up=1, eta=self.ETAC)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                         low=0, up=1, eta=self.ETAM, indpb=self.MIDPB)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("map", futures.map)
        self.toolbox = toolbox

    def evolve(self, npop, ngen, seed=None, pre=''):
        """ Generate and evolve the population based on NSGA-II.

        Keyword arguments:
        npop -- population size or list of existed populations.
        ngen -- number of generations.
        seed -- [None] the random seed.
        pre -- [''] relative path of the root of the simulation folders.
        """
        init = 1  # flag show that if it's a new run
        if isinstance(npop, int):
            if npop % 4:
                raise OptimizerError('population size has to be a multiple of 4!')
            random.seed(seed)
        else:
            init = 0
            ipop = npop
            npop = len(npop)
        toolbox = self.toolbox

        def ind_fitness(ind):
            return ind.fitness.values

        stats = tools.Statistics(ind_fitness)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Begin the generational process
        for gen in tqdm(range(ngen), desc='Generation', ascii=True):
            if not gen:
                pop = []
                if init:
                    offspring = toolbox.population(n=npop)
                else:
                    offspring = ipop
            else:
                # Vary the population
                offspring = tools.selTournamentDCD(pop, npop)
                offspring = [toolbox.clone(ind) for ind in offspring]

                for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() <= self.CXPB: toolbox.mate(ind1, ind2)
                    if random.random() <= self.MPB: toolbox.mutate(ind1)
                    if random.random() <= self.MPB: toolbox.mutate(ind2)
                    del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            sims = [os.path.join(pre, '{0:03d}'.format(i + 1)) for i in range(len(invalid_ind))]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, sims)
            with tqdm(total=len(invalid_ind), desc='Population', ascii=True) as pbar:
                for ind, fit in zip(invalid_ind, fitnesses):
                    pbar.update(1)
                    ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, npop)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            # print(logbook.stream)
            # Save the populations of the latest generation
            with open('pop', 'wb') as f:
                pickle.dump([gen, pop], f)
        # print("Done!")

        self.pop = pop
        self.log = logbook


class SPEA2:
    """ SPEA2 optimizer.
    """

    def __init__(self, evaluate=None):
        """ Initialize the optimizer.

        Keyword arguments:
        evaluate -- [None] the evaluate function.
        """
        self.NDIM = 30
        self.OBJ = (-1.0, -1.0)
        self.ETAC = 1.0
        self.ETAM = 1.0
        self.CXPB = 0.9
        self.MPB = 0.3
        self.MIDPB = 0.5
        self.evaluate = evaluate
        self.toolbox = None
        self.pop = None
        self.log = None
        self.setup()

    def setup(self):
        """ Call this method when any member variable changes to update
        the SPEA2 arguments.
        """
        try:
            del creator.Quality
        except AttributeError:
            pass
        try:
            del creator.Individual
        except AttributeError:
            pass
        creator.create("Quality", base.Fitness, weights=self.OBJ)
        creator.create("Individual", list, fitness=creator.Quality)

        toolbox = base.Toolbox()

        toolbox.register("norm_var", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.norm_var, self.NDIM)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # toolbox.register("evaluate", benchmarks.zdt2)
        if not self.evaluate:
            warnings.warn("evaluate function not defined!", OptimizerWarning)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                         low=0, up=1, eta=self.ETAC)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                         low=0, up=1, eta=self.ETAM, indpb=self.MIDPB)
        toolbox.register("select", tools.selSPEA2)
        toolbox.register("selectTournament", tools.selTournament, tournsize=2)
        toolbox.register("map", futures.map)
        self.toolbox = toolbox

    def evolve(self, npop, narc, ngen, seed=None, pre=''):
        """ Generate and evolve the population based on SPEA2.

        Keyword arguments:
        npop -- population size or list of existed populations.
        ngen -- number of generations.
        narc -- capacity of archive.
        seed -- [None] the random seed.
        pre -- [''] relative path of the root of the simulation folders.
        """
        init = 1  # flag show that if it's a new run
        if isinstance(npop, int):
            random.seed(seed)
        else:
            init = 0
            ipop = npop
            npop = len(npop)
        toolbox = self.toolbox

        def ind_fitness(ind):
            return ind.fitness.values

        stats = tools.Statistics(ind_fitness)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Step 1 Initialization
        if init:
            pop = toolbox.population(n=npop)
        else:
            pop = ipop
        archive = []

        for gen in tqdm(range(ngen), leave=True, ascii=True):
            # Step 2 Fitness assignment
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            sims = [os.path.join(pre, '{0:03d}'.format(i + 1)) for i in range(len(invalid_ind))]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, sims)
            with tqdm(total=len(invalid_ind), desc='Population', ascii=True) as pbar:
                for ind, fit in zip(invalid_ind, fitnesses):
                    pbar.update(1)
                    ind.fitness.values = fit

            # Step 3 Environmental selection
            archive = toolbox.select(pop + archive, k=narc)

            # Step 5 Mating selection
            offspring = toolbox.selectTournament(archive, k=npop)
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Step 6 Variation
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.CXPB: toolbox.mate(ind1, ind2)
                if random.random() <= self.MPB: toolbox.mutate(ind1)
                if random.random() <= self.MPB: toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            pop = offspring
            record = stats.compile(archive)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            # print(logbook.stream)
            # Save the populations of the latest generation
            with open('pop', 'wb') as f:
                pickle.dump([gen, archive], f)
        # print("Done!")

        self.pop = archive
        self.log = logbook
