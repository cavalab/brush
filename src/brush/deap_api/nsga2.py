from deap import tools 
from deap.benchmarks.tools import diversity, convergence, hypervolume
import numpy as np
import random

def nsga2(toolbox, NGEN, MU, CXPB, verbosity):
    # NGEN = 250
    # MU = 100
    # CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("ave", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    # stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "offspring", "ave", "std", "min" 

    pop = toolbox.population(n=MU)


    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.survive(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if verbosity > 0: 
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        # offspring = tools.selTournamentDCD(pop, len(pop))
        parents = toolbox.select(pop, len(pop))
        # offspring = [toolbox.clone(ind) for ind in offspring]
        offspring = []

        for ind1, ind2 in zip(parents[::2], parents[1::2]):
            if random.random() <= CXPB:
                off1, off2 = toolbox.mate(ind1, ind2)
            else:
                off1, off2 = ind1, ind2
                
            # avoid inserting empty solutions
            if off1: off1 = toolbox.mutate(off1)
            if off1: offspring.extend([off1])

            if off2: off2 = toolbox.mutate(off2)
            if off2: offspring.extend([off2])

        # archive.update(offspring)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.survive(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), offspring=len(offspring), **record)
        if verbosity > 0: 
            print(logbook.stream)

    if verbosity > 0: 
        print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 50.0]))

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook