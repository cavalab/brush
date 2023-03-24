from deap import tools 
from deap.benchmarks.tools import diversity, convergence, hypervolume
import numpy as np
import random

def nsga2(toolbox, NGEN, MU, CXPB):
    # NGEN = 250
    # MU = 100
    # CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("ave", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    # stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "ave", "std", "min" 

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
                ind1, ind2 = toolbox.mate(ind1, ind2)

            off1 = toolbox.mutate(ind1)
            off2 = toolbox.mutate(ind2)
            # del ind1.fitness.values, ind2.fitness.values
            offspring.extend([off2, off2])

        # archive.update(offspring)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.survive(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 50.0]))

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook