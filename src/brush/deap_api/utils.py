import numpy as np


class DeapIndividual():
    """Class that wraps brush program for creator.Individual class from DEAP."""
    def __init__(self, prg):
        self.prg = prg


def e_lexicase(individuals, k):
    """
    Based on DEAP implementation (tools.selAutomaticEpsilonLexicase), but
    working on individual predictions instead of different fitness cases.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """

    selected_individuals = []
    for _ in range(k):
        candidates = individuals
        cases = [i for i in range(len(individuals[0].errors))]
        np.random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_on_case = np.array([x.errors[cases[0]] for x in candidates])
            
            MAD = np.median(np.abs(errors_on_case - np.median(errors_on_case)))
            best_on_case = np.min(errors_on_case)

            # Skip if this case is np.inf for all individuals
            if not np.isfinite(best_on_case+MAD):
                continue
            
            candidates = [x for x in candidates
                          if x.errors[cases[0]] <= best_on_case + MAD]

            cases.pop(0)

        selected_individuals.append(np.random.choice(candidates))

    return selected_individuals