"""
sklearn-compatible wrapper for GP analyses.

See brushgp.cpp for Python (via pybind11) modules that give more fine-grained
control of the underlying GP objects.
"""
# from sklearn.base import BaseEstimator
# from sklearn.metrics import mean_squared_error
import numpy as np
# import deap as dp
from deap import algorithms, base, creator, tools
# from tqdm import tqdm

import _brush
# from _brush import Dataset, SearchSpace

class Fitness():
    def __init__(self):
        self.values = None
        self.valid = False

class Individual():
    """Class that wraps brush program for creator.Individual class from DEAP.
    """
    def __init__(self, prg):
        self.prg = prg
        self.fitness = Fitness()

class BrushEstimator():
    """
    Binary classifier using a GP tree.

    Parameters
    ----------
    max_depth : int, default 0
        Maximum depth of GP trees in the GP program. Use 0 for no limit.
    max_breadth : int, default 0
        Maximum width of the tree at its widest point. Use 0 for no limit.
    max_size : int, default 0
        Maximum number of nodes in a tree. Use 0 for no limit.
    """
    
    def __init__(
        self, 
        mode='classification',
        pop_size=100,
        max_gen=100,
        verbosity=0,
        max_depth=3,
        max_size=20
        ):
        self.pop_size=pop_size
        self.max_gen=max_gen
        self.verbosity=verbosity
        self.mode=mode
        self.max_depth=max_depth
        self.max_size=max_size



    def _setup_toolbox(self, data):
        """Setup the deap toolbox"""
        toolbox: base.Toolbox = base.Toolbox()

        creator.create("FitnessMulti", base.Fitness, weights=(1.0,-1.0))
        
        # creator.create("Individual", self.Individual, fitness=creator.FitnessMulti)  
        # self._create_deap_individual_ = creator.Individual

        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate)

        toolbox.register("select", tools.selNSGA2, nd='log')

        toolbox.register(
            "evaluate",
            self._fitness_function,
            data=data
        )

        return toolbox

    def _crossover(self, ind1, ind2):
        return (
            Individual(ind1.prg.cross(ind2.prg)), 
            Individual(ind2.prg.cross(ind1.prg))
            )

    def _mutate(self, ind1):
        return (Individual(ind1.prg.mutate(self.search_space_)),)

    def fit(self, X, y):
        """
        Fit an estimator to X,y.

        Parameters
        ----------
        X : np.ndarray
            2-d array of input data.
        y : np.ndarray
            1-d array of (boolean) target values.
        """

        self.data_ = _brush.Dataset(X, y)
        self.search_space_ = _brush.SearchSpace(self.data_)
        self.hof_ = tools.HallOfFame(maxsize=self.pop_size)
        self.toolbox_ = self._setup_toolbox(data=self.data_)
        population = self._setup_population()
        # This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        # :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        # registered in the toolbox. This algorithm uses the :func:`varOr`
        # variation.
        algorithms.eaMuPlusLambda(
            population=population,
            toolbox=self.toolbox_,
            # halloffame=self.hof_,
            mu=self.pop_size,
            lambda_=self.pop_size,
            ngen=self.max_gen,
            cxpb=0.5,
            mutpb=0.5
        )

        return self

    def _setup_population(self):
        """initialize programs"""
        if self.mode == 'classification':
            generate = self.search_space_.make_classifier
        else:
            generate = self.search_space_.make_regressor

        programs = [
            Individual(generate(
                np.random.randint(1,self.max_depth),
                np.random.randint(1, self.max_size)
                ))
            for i in range(self.pop_size)
        ]
        # return [self._create_deap_individual_(p) for p in programs]
        return programs

class BrushClassifier(BrushEstimator):
    """Brush for classification"""
    def __init__( self, **kwargs):
        super().__init__(mode='classification',**kwargs)

    def _get_program_type(self):
        return _brush.program.Classifier

    def _fitness_function(self, ind, data: _brush.Dataset):
        ind.prg.fit(data)
        ind.prg.set_search_space(self.search_space_)
        return (
            np.abs(data.y-ind.prg.predict(data)).sum(), 
            ind.prg.size()
        )
    # class Individual(_brush.program.Classifier):
    #     """Class that wraps brush program for creator.Individual class from DEAP. """
    #     def __init__(self,*args, **kwargs):
    #         print('args:',args)
    #         print('kwargs:',kwargs)
    #         super().__init__()

class BrushRegressor(BrushEstimator):
    """Brush for classification"""
    def __init__( self, **kwargs):
        super().__init__(mode='regressor',**kwargs)

    def _get_program_type(self):
        return _brush.program.Regressor
    
    def _fitness_function(self, ind, data: _brush.Dataset):
        return (
            np.sum((data.y- ind.predict(data))**2),
            ind.size()
        )
