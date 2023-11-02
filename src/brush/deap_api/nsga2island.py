from deap import tools 
from deap.benchmarks.tools import diversity, convergence, hypervolume
import numpy as np
import functools


def nsga2island(toolbox, NGEN, MU, N_ISLANDS, MIGPX, CXPB, use_batch, verbosity, rnd_flt):
    # NGEN = 250
    # MU   = 100
    # CXPB = 0.9
    # N_ISLANDS: number of independent islands. Islands are controled by indexes.
    # setting N_ISLANDS=1 would be the same as the original nsga2
    # rnd_flt: random number generator to sample crossover prob

    def calculate_statistics(ind):
        on_train = ind.fitness.values
        on_val   = toolbox.evaluateValidation(ind)

        return (*on_train, *on_val) 

    stats = tools.Statistics(calculate_statistics)

    stats.register("avg", np.mean, axis=0)
    stats.register("med", np.median, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + \
                     [f"{stat} {partition} O{objective}"
                         for stat in ['avg', 'med', 'std', 'min', 'max']
                         for partition in ['train', 'val']
                         for objective in toolbox.get_objectives()]

    # Tuples with start and end indexes for each island. Number of individuals
    # in each island can slightly differ if N_ISLANDS is not a divisor of MU
    island_indexes = [((i*MU)//N_ISLANDS, ((i+1)*MU)//N_ISLANDS)
                      for i in range(N_ISLANDS)]

    pop = toolbox.population(n=MU)

    fitnesses = toolbox.map(functools.partial(toolbox.evaluate), pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    survived = []
    for (idx_start, idx_end) in island_indexes:
        survived_parents = toolbox.survive(pop[idx_start:idx_end],
                                        idx_end-idx_start)
        survived.extend(survived_parents)
    pop = survived

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)

    if verbosity > 0: 
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        batch = toolbox.getBatch() # batch will be a random subset only if it was not
                                   # defined as the size of the train set. everytime
                                   # this function is called, a new random batch is generated.

        if (use_batch): # recalculate the fitness for the parents
            # use_batch is false if batch_size is different from train set size.
            # If we're using batch, we need to re-evaluate every model (without
            # changing its weights). evaluateValidation doesnt fit the weights
            fitnesses = toolbox.map( 
                functools.partial(toolbox.evaluateValidation, data=batch), pop)
        
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

        # Vary the population inside each island
        parents = []
        for (idx_start, idx_end) in island_indexes:
            island_parents = toolbox.select(pop[idx_start:idx_end],
                                            idx_end-idx_start)
            parents.extend(island_parents)
    
        offspring = [] # Will have the same size as pop
        island_failed_variations = []
        for (idx_start, idx_end) in island_indexes:
            failed_variations = 0
            for ind1, ind2 in zip(parents[idx_start:idx_end:2],
                                  parents[idx_start+1:idx_end:2]
            ):
                off1, off2 = None, None
                if rnd_flt() < CXPB: # either mutation or crossover
                    off1, off2 = toolbox.mate(ind1, ind2)
                else:
                    off1 = toolbox.mutate(ind1)
                    off2 = toolbox.mutate(ind2)
                
                if off1 is not None:
                    off1.fitness.values = toolbox.evaluate(off1) 
                    if use_batch:
                        off1.fitness.values = toolbox.evaluateValidation(off1, data=batch)
                    offspring.extend([off1])
                else:
                    failed_variations += 1

                if off2 is not None:
                    off2.fitness.values = toolbox.evaluate(off2) 
                    if use_batch:
                        off2.fitness.values = toolbox.evaluateValidation(off2, data=batch)
                    offspring.extend([off2])
                else:
                    failed_variations += 1

        # Evaluate (instead of evaluateValidation) to fit the weights of the offspring
        fitnesses = toolbox.map(functools.partial(toolbox.evaluate), offspring)
        if (use_batch): #calculating objectives based on batch
            fitnesses = toolbox.map(
                functools.partial(toolbox.evaluateValidation, data=batch), offspring)

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        new_pop = []
        for i, (idx_start, idx_end) in enumerate(island_indexes):
            # original population combined with offspring, taking into account that variations can fail
            island_new_pop = toolbox.survive(
                pop[idx_start:idx_end] \
                + offspring[
                    idx_start-sum(island_failed_variations[:i]):idx_end+island_failed_variations[i]
                ], 
                idx_end-idx_start # number of selected individuals should still the same
            )
            new_pop.extend(island_new_pop)

        # Migration to fill up the islands for the next generation
        pop = []
        for (idx_start, idx_end) in island_indexes:
            other_islands = list(range(0, idx_start)) + list(range(idx_end, MU))
            for idx_individual in range(idx_start, idx_end):
                if rnd_flt() < MIGPX: # replace by someone not from the same island
                    idx_other_individual = other_islands[
                        int(rnd_flt() * len(other_islands))]
                    pop.append(new_pop[idx_other_individual])
                else:
                    pop.append(new_pop[idx_individual])
                    
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring)+(len(pop) if use_batch else 0), **record)

        if verbosity > 0: 
            print(logbook.stream)

    if verbosity > 0: 
        print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 50.0]))

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook