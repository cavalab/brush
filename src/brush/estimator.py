"""
sklearn-compatible wrapper for GP analyses.

See brushgp.cpp for Python (via pybind11) modules that give more fine-grained
control of the underlying GP objects.
"""
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
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
    mode : str, default 'classification'
        The mode of the estimator. Used by subclasses
    pop_size : int, default 100
        Population size.
    max_gen : int, default 100
        Maximum iterations of the algorithm.
    verbosity : int, default 0
        Controls level of printouts.
    max_depth : int, default 0
        Maximum depth of GP trees in the GP program. Use 0 for no limit.
    max_size : int, default 0
        Maximum number of nodes in a tree. Use 0 for no limit.
    cx_prob : float, default 0.9
        Probability of applying the crossover variation when generating the offspring
    mutation_options : dict, default {"point":0.4, "insert":0.25, "delete":0.25, "toggle_weight":0.1}
        A dictionary with keys naming the types of mutation and floating point 
        values specifying the fraction of total mutations to do with that method. 
    functions: dict[str,float] or list[str], default {}
        A dictionary with keys naming the function set and values giving the probability of sampling them, or a list of functions which will be weighted uniformly.
        If empty, all available functions are included in the search space.

    Attributes
    ----------
    best_estimator_ : _brush.Program
        The final model picked from training. Used in subsequent calls to :func:`predict`. 
    archive_ : list[deap_api.DeapIndividual]
        The final population from training. 
    data_ : _brush.Dataset
        The training data in Brush format. 
    search_space_ : a Brush `SearchSpace` object. 
        Holds the operators and terminals and sampling utilities to update programs.
    toolbox_ : deap.Toolbox
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
        cx_prob=0.9,
        mutation_options = {"point":0.4, "insert":0.25, "delete":0.25, "toggle_weight":0.1},
        functions: list[str]|dict[str,float] = {},
        batch_size: int = 0
        ):
        self.pop_size=pop_size
        self.max_gen=max_gen
        self.verbosity=verbosity
        self.mode=mode
        self.max_depth=max_depth
        self.max_size=max_size
        self.cx_prob=cx_prob
        self.mutation_options=mutation_options
        self.functions=functions
        self.batch_size=batch_size


    def _setup_toolbox(self, data):
        """Setup the deap toolbox"""
        toolbox: base.Toolbox = base.Toolbox()

        # creator.create is used to "create new functions", and takes at least
        # 2 arguments: the name of the newly created class and a base class

        # Minimizing/maximizing problem: negative/positive weight, respectively.
        # Our classification is using the error as a metric
        # Comparing fitnesses: https://deap.readthedocs.io/en/master/api/base.html#deap.base.Fitness
        creator.create("FitnessMulti", base.Fitness, weights=(+1.0,-1.0))

        # create Individual class, inheriting from self.Individual with a fitness attribute
        creator.create("Individual", DeapIndividual, fitness=creator.FitnessMulti)  
        
        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate)

        # When solving multi-objective problems, selection and survival must
        # support this feature. This means that these selection operators must
        # accept a tuple of fitnesses as argument)
        toolbox.register("select", tools.selTournamentDCD) 
        toolbox.register("survive", tools.selNSGA2)

        # toolbox.population will return a list of elements by calling toolbox.individual
        toolbox.register("population", tools.initRepeat, list, self._make_individual)
        toolbox.register( "evaluate", self._fitness_function, data=data)

        return toolbox

    def _crossover(self, ind1, ind2):
        offspring = [] 

        for i,j in [(ind1,ind2),(ind2,ind1)]:
            child = i.prg.cross(j.prg)
            if child:
                offspring.append(creator.Individual(child))
            else: # so we'll always have two elements to unpack in `offspring`
                offspring.append(None)

        return offspring[0], offspring[1]

    def _mutate(self, ind1):
        # offspring = (creator.Individual(ind1.prg.mutate(self.search_space_)),)
        offspring = ind1.prg.mutate()
        
        if offspring:
            return creator.Individual(offspring)
        
        return None

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

        # set n classes if relevant
        if self.mode=="classification":
            self.n_classes_ = len(np.unique(y))

        if isinstance(self.functions, list):
            self.functions_ = {k:1.0 for k in self.functions}
        else:
            self.functions_ = self.functions

        self.search_space_ = _brush.SearchSpace(self.data_, self.functions_)
        self.toolbox_ = self._setup_toolbox(data=self.data_)

        archive, logbook = nsga2(self.toolbox_, self.max_gen, self.pop_size, self.cx_prob, self.verbosity)
        self.archive_ = archive
        self.logbook_ = logbook
        self.best_estimator_ = self.archive_[0].prg

        if self.verbosity > 0:
            print(f'best model {self.best_estimator_.get_model()}'+
                  f' with size {self.best_estimator_.size()}, '   +
                  f' depth {self.best_estimator_.depth()}, '      +
                  f' and fitness {self.archive_[0].fitness}'      )

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

        assert isinstance(X, np.ndarray)
        # if there is no label, don't include it in library call to Dataset
        if isinstance(y, NoneType):
            return _brush.Dataset(X)

        return _brush.Dataset(X,y)

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
        return ( # (accuracy, size)
            (data.y==ind.prg.predict(data)).sum() / data.y.shape[0], 
            ind.prg.size()
        )
    
    def _make_individual(self):
        # C++'s PTC2-based `make_individual` will create a tree of at least
        # the given size. By uniformly sampling the size, we can instantiate a
        # population with more diversity
        s = np.random.randint(1, self.max_size)

        return creator.Individual(
            self.search_space_.make_classifier(self.max_depth, s)
            if self.n_classes_ == 2 else
            self.search_space_.make_multiclass_classifier(self.max_depth, s)
            )

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        """
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

        MSE = np.mean( (data.y-ind.prg.predict(data))**2 )
        if not np.isfinite(MSE): # numeric erros, np.nan, +-np.inf
            MSE = np.inf

        # We are squash the error and making it a maximization problem
        return ( 1/(1+MSE), ind.prg.size() )

    def _make_individual(self):
        s = np.random.randint(1, self.max_size)
        
        return creator.Individual(
            self.search_space_.make_regressor(self.max_depth, s)
        )

# Under development
# class BrushRepresenter(BrushEstimator, TransformerMixin):
#     """Brush for representation learning.

#     For options, see :py:class:`BrushEstimator <brush.estimator.BrushEstimator>`. 

#     Examples
#     --------
#     >>> import pandas as pd
#     >>> df = pd.read_csv('docs/examples/datasets/d_enc.csv')
#     >>> X = df.drop(columns='label')
#     >>> y = df['label']
#     >>> from brush import BrushRegressor
#     >>> est = BrushRegressor()
#     >>> est.fit(X,y)
#     >>> print('score:', est.score(X,y))
#     """
#     def __init__(self, **kwargs):
#         super().__init__(mode='regressor',**kwargs)

#     def _fitness_function(self, ind, data: _brush.Dataset):
#         ind.prg.fit(data)
#         return (
#             # todo: need to return a matrix from X for this
#             np.sum((data.get_X()- ind.prg.predict(data))**2),
#             ind.prg.size()
#         )

#     def _make_individual(self):
#         return creator.Individual(
#             self.search_space_.make_representer(self.max_depth, self.max_size)
#         )

#     def transform(self, X):
#         """Transform X using the best estimator in the archive. """
#         return self.predict(X)