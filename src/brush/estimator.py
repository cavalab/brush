"""
sklearn-compatible wrapper for GP analyses.

See brushgp.cpp for Python (via pybind11) modules that give more fine-grained
control of the underlying GP objects.
"""
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
# from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
# import deap as dp
from deap import algorithms, base, creator, tools
# from tqdm import tqdm
from types import NoneType
import _brush
from .deap_api import nsga2, DeapIndividual 
# from _brush import Dataset, SearchSpace


class BrushEstimator(BaseEstimator):
    """
    This is the base class for Brush estimators. 
    This class shouldn't be called directly; instead, call a child class like 
    :py:class:`BrushRegressor <brush.estimator.BrushRegressor>` or :py:class:`BrushClassifier <brush.estimator.BrushClassifier>`. 
    All of the shared parameters are documented here. 
    

    Parameters
    ----------
    mode: str, default 'classification'
        The mode of the estimator. Used by subclasses
    pop_size: int, default 100
        Population size.
    max_gen: int, default 100
        Maximum iterations of the algorithm.
    verbosity: int, default 0
        Controls level of printouts.
    max_depth : int, default 0
        Maximum depth of GP trees in the GP program. Use 0 for no limit.
    max_size : int, default 0
        Maximum number of nodes in a tree. Use 0 for no limit.
    mutation_options: dict, default {"point":0.5, "insert": 0.25, "delete":  0.25}
        A dictionary with keys naming the types of mutation and floating point 
        values specifying the fraction of total mutations to do with that method. 

    Attributes
    ----------
    best_estimator_: _brush.Program
        The final model picked from training. Used in subsequent calls to :func:`predict`. 
    archive_: list[deap_api.DeapIndividual]
        The final population from training. 
    data_: _brush.Dataset
        The training data in Brush format. 
    search_space_: a Brush `SearchSpace` object. 
        Holds the operators and terminals and sampling utilities to update programs.
    toolbox_: deap.Toolbox
        The toolbox used by DEAP for EA algorithm. 

    """
    
    def __init__(
        self, 
        mode='classification',
        pop_size=100,
        max_gen=100,
        verbosity=0,
        max_depth=3,
        max_size=20,
        mutation_options = {"point":0.5, "insert": 0.25, "delete":  0.25},
        ):
        self.pop_size=pop_size
        self.max_gen=max_gen
        self.verbosity=verbosity
        self.mode=mode
        self.max_depth=max_depth
        self.max_size=max_size
        self.mutation_options=mutation_options


    def _setup_toolbox(self, data):
        """Setup the deap toolbox"""
        toolbox: base.Toolbox = base.Toolbox()

        # minimize MAE, minimize size
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0))

        # create Individual class, inheriting from self.Individual with a fitness attribute
        creator.create("Individual", DeapIndividual, fitness=creator.FitnessMulti)  
        
        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate)

        toolbox.register("select", tools.selTournamentDCD) 
        toolbox.register("survive", tools.selNSGA2)

        # toolbox.individual will make an individual by calling self._make_individual
        # toolbox.register("individual", creator.Individual, self._make_individual)
        # toolbox.population will return a list of elements by calling toolbox.individual
        toolbox.register("population", tools.initRepeat, list, self._make_individual)
        toolbox.register( "evaluate", self._fitness_function, data=data)

        return toolbox

    def _crossover(self, ind1, ind2):
        offspring = [] 

        for i,j in [(ind1,ind2),(ind2,ind1)]:
            off = creator.Individual(i.prg.cross(j.prg))
            # off.fitness.valid = False
            offspring.append(off)

        return offspring[0], offspring[1]

    def _mutate(self, ind1):
        # offspring = (creator.Individual(ind1.prg.mutate(self.search_space_)),)
        offspring = creator.Individual(ind1.prg.mutate(self.search_space_))
        return offspring

    # def _set_brush_params(self, attribs):
    #     for k,v in attribs.items():
    #     _brush.PARAMS = attribs

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
        _brush.set_params(self.get_params())
        self.data_ = self._make_data(X,y)
        self.search_space_ = _brush.SearchSpace(self.data_)
        # self.hof_ = tools.HallOfFame(maxsize=self.pop_size)
        self.toolbox_ = self._setup_toolbox(data=self.data_)

        archive, logbook = nsga2(self.toolbox_, self.max_gen, self.pop_size, 0.9)
        self.archive_ = archive
        self.best_estimator_ = self.archive_[0].prg

        print('best model:',self.best_estimator_.get_model())
        return self
    
    def _make_data(self, X, y=None):
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(X, pd.DataFrame):
            # self.data_ = _brush.Dataset(X.to_dict(orient='list'), y)
            feature_names = X.columns.to_list()
            X = X.values
            if isinstance(y, NoneType):
                return _brush.Dataset(X, feature_names)
            else:
                return _brush.Dataset(X, y, feature_names)
        else:
            assert isinstance(X, np.ndarray)
        if isinstance(y, NoneType):
            return _brush.Dataset(X, y)
        return _brush.Dataset(X)

    def predict(self, X):
        """Predict using the best estimator in the archive. """
        data = self._make_data(X)
        return self.best_estimator_.predict(data)

    # def _setup_population(self):
    #     """initialize programs"""
    #     if self.mode == 'classification':
    #         generate = self.search_space_.make_classifier
    #     else:
    #         generate = self.search_space_.make_regressor

    #     programs = [
    #         DeapIndividual(generate(self.max_depth, self.max_size))
    #         for i in range(self.pop_size)
    #     ]
    #     # return [self._create_deap_individual_(p) for p in programs]
    #     return programs

    def get_params(self):
        return {k:v for k,v in self.__dict__.items() if not k.endswith('_')}

class BrushClassifier(BrushEstimator,ClassifierMixin):
    """Brush for classification.

    For options, see :py:class:`BrushEstimator <brush.estimator.BrushEstimator>`. 

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    >>> X = df.drop(columns='target')
    >>> y = df['target']
    >>> from brush import BrushClassifier
    >>> est = BrushClassifier()
    >>> est.fit(X,y)
    >>> print('score:', est.score(X,y))
    """
    def __init__( self, **kwargs):
        super().__init__(mode='classification',**kwargs)

    def _fitness_function(self, ind, data: _brush.Dataset):
        ind.prg.fit(data)
        return (
            np.abs(data.y-ind.prg.predict(data)).sum(), 
            ind.prg.size()
        )
    
    def _make_individual(self):
        return creator.Individual(
            self.search_space_.make_classifier(self.max_depth, self.max_size)
            )

    def predict_proba(self, X):
        """Predict using the best estimator in the archive. """
        data = self._make_data(X)
        return self.best_estimator_.predict_proba(data)

class BrushRegressor(BrushEstimator, RegressorMixin):
    """Brush for regression.

    For options, see :py:class:`BrushEstimator <brush.estimator.BrushEstimator>`. 

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_enc.csv')
    >>> X = df.drop(columns='label')
    >>> y = df['label']
    >>> from brush import BrushRegressor
    >>> est = BrushRegressor()
    >>> est.fit(X,y)
    >>> print('score:', est.score(X,y))
    """
    def __init__(self, **kwargs):
        super().__init__(mode='regressor',**kwargs)

    def _fitness_function(self, ind, data: _brush.Dataset):
        ind.prg.fit(data)
        return (
            np.sum((data.y- ind.prg.predict(data))**2),
            ind.prg.size()
        )

    def _make_individual(self):
        return creator.Individual(
            self.search_space_.make_regressor(self.max_depth, self.max_size)
        )