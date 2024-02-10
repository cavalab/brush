from deap import tools 
from deap.benchmarks.tools import hypervolume
import numpy as np
import functools

class DeapIndividual():
    """Class that wraps brush program for creator.Individual class from DEAP."""
    def __init__(self, program):
        self.program = program

def nsga2(toolbox, NGEN, MU, CXPB, use_batch, verbosity, rnd_flt):
    # NGEN = 250
    # MU   = 100
    # CXPB = 0.9
    # rnd_flt: random number generator to sample crossover prob

    def calculate_statistics(ind):
        on_train = ind.fitness.values
        # TODO: make this work again
        on_val   = ind.fitness.values #toolbox.evaluateValidation(ind)

        return (*on_train, *on_val) 

    stats = tools.Statistics(calculate_statistics)

    stats.register("avg", np.nanmean, axis=0)
    stats.register("med", np.nanmedian, axis=0)
    stats.register("std", np.nanstd, axis=0)
    stats.register("min", np.nanmin, axis=0)
    stats.register("max", np.nanmax, axis=0)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + \
                     [f"{stat} {partition} O{objective}"
                         for stat in ['avg', 'med', 'std', 'min', 'max']
                         for partition in ['train', 'val']
                         for objective in toolbox.get_objectives()]

    pop = toolbox.population(n=MU)
    pop = list(toolbox.map(toolbox.assign_fit, pop))

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.survive(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)

    if verbosity > 0: 
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        parents = toolbox.select(pop, len(pop))
        offspring = []
        for ind1, ind2 in zip(parents, parents[1:]):
            off = None
            if rnd_flt() < CXPB: # either mutation or crossover
                off = toolbox.mate(ind1, ind2)
            else:
                off = toolbox.mutate(ind1)
            
            if off is not None: # first we fit, then add to offspring
                offspring.extend([off])

        offspring = list(toolbox.map(toolbox.assign_fit, offspring))
        # Select the next generation population (no sorting before this step, as 
        # survive==offspring will cut it in half)
        pop = toolbox.survive(pop + offspring, MU)

        pop.sort(key=lambda x: x.fitness, reverse=True)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring)+(len(pop) if use_batch else 0), **record)

        if verbosity > 0: 
            print(logbook.stream)
            print(pop[0].fitness.values, pop[0].fitness.weights, pop[0].fitness.wvalues,
                  pop[0].program.get_model(),)

    # if verbosity > 0: 
    #     print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 50.0]))

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook