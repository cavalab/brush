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
    cx_prob : float, default 1/7
        Probability of applying the crossover variation when generating the offspring,
        must be between 0 and 1.
        Given that there are `n` mutations, and either crossover or mutation is 
        used to generate each individual in the offspring (but not both at the
        same time), we want to have by default an uniform probability between
        crossover and every possible mutation. By setting `cx_prob=1/(n+1)`, and
        `1/n` for each mutation, we can achieve an uniform distribution.
    mutation_options : dict, default {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6, "toggle_weight_on":1/6, "toggle_weight_off":1/6}
        A dictionary with keys naming the types of mutation and floating point 
        values specifying the fraction of total mutations to do with that method.
        The probability of having a mutation is `(1-cx_prob)` and, in case the mutation
        is applied, then each mutation option is sampled based on the probabilities
        defined in `mutation_options`. The set of probabilities should add up to 1.0.
    functions: dict[str,float] or list[str], default {}
        A dictionary with keys naming the function set and values giving the probability
        of sampling them, or a list of functions which will be weighted uniformly.
        If empty, all available functions are included in the search space.
    initialization : {"grow", "full"}, default "grow" 
        Strategy to create the initial population. If `full`, then every expression is created
        with `max_size` nodes. If `grow`, size will be uniformly distributed.
    algorithm : {"nsga2", "ga"}, default "nsga2"
        Which Evolutionary Algorithm framework to use to evolve the population.
    validation_size : float, default 0.0
        Percentage of samples to use as a hold-out partition. These samples are used
        to calculate statistics during evolution, but not used to train the models.
        The `best_estimator_` will be selected using this partition. If zero, then
        the same data used for training is used for validation.
    batch_size : float, default 1.0
        Percentage of training data to sample every generation. If `1.0`, then
        all data is used. Very small values can improve execution time, but 
        also lead to underfit.
    random_state: int or None, default None
        If int, then the value is used to seed the c++ random generator; if None,
        then a seed will be generated using a non-deterministic generator. It is
        important to notice that, even if the random state is fixed, it is
        unlikely that running brush using multiple threads will have the same
        results. This happens because the Operating System's scheduler is
        responsible to choose which thread will run at any given time, thus 
        reproductibility is not guaranteed.

    Attributes
    ----------
    best_estimator_ : _brush.Program
        The final model picked from training. Used in subsequent calls to :func:`predict`. 
    archive_ : list[deap_api.DeapIndividual]
        The final population from training. 
    data_ : _brush.Dataset
        The complete data in Brush format. 
    train_ : _brush.Dataset
        Partition of `data_` containing `(1-validation_size)`% of the data, in Brush format.
    validation_ : _brush.Dataset
        Partition of `data_` containing `(validation_size)`% of the data, in Brush format.
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
        cx_prob= 1/7,
        mutation_options = {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6,
                            "toggle_weight_on":1/6, "toggle_weight_off":1/6},
        functions: list[str]|dict[str,float] = {},
        initialization="grow",
        algorithm="nsga2",
        random_state=None,
        validation_size: float = 0.0,
        batch_size: float = 1.0
        ):
        self.pop_size=pop_size
        self.max_gen=max_gen
        self.verbosity=verbosity
        self.algorithm=algorithm
        self.mode=mode
        self.max_depth=max_depth
        self.max_size=max_size
        self.cx_prob=cx_prob
        self.mutation_options=mutation_options
        self.functions=functions
        self.initialization=initialization
        self.random_state=random_state
        self.batch_size=batch_size
        self.validation_size=validation_size


    def _setup_toolbox(self, data_train, data_validation):
        """Setup the deap toolbox"""
        toolbox: base.Toolbox = base.Toolbox()

        # creator.create is used to "create new functions", and takes at least
        # 2 arguments: the name of the newly created class and a base class

        # Minimizing/maximizing problem: negative/positive weight, respectively.
        # Our classification is using the error as a metric
        # Comparing fitnesses: https://deap.readthedocs.io/en/master/api/base.html#deap.base.Fitness
        creator.create("FitnessMulti", base.Fitness, weights=self.weights)

        # create Individual class, inheriting from self.Individual with a fitness attribute
        creator.create("Individual", DeapIndividual, fitness=creator.FitnessMulti)  

        toolbox.register("Clone", lambda ind: creator.Individual(ind.prg.copy()))
        
        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate)

        # When solving multi-objective problems, selection and survival must
        # support this feature. This means that these selection operators must
        # accept a tuple of fitnesses as argument)
        if self.algorithm=="nsga2":
            toolbox.register("select", tools.selTournamentDCD) 
            toolbox.register("survive", tools.selNSGA2)
        elif self.algorithm=="ga":
            toolbox.register("select", tools.selTournament, tournsize=3) 
            def offspring(pop, MU): return pop[-MU:]
            toolbox.register("survive", offspring)

        # toolbox.population will return a list of elements by calling toolbox.individual
        toolbox.register("createRandom", self._make_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.createRandom)

        toolbox.register("getBatch", data_train.get_batch)
        toolbox.register("evaluate", self._fitness_function, data=data_train)
        toolbox.register("evaluateValidation", self._fitness_validation, data=data_validation)

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
        
        if self.random_state is not None:
            _brush.set_random_state(self.random_state)

        self.data_ = self._make_data(X,y, validation_size=self.validation_size)

        # set n classes if relevant
        if self.mode=="classification":
            self.n_classes_ = len(np.unique(y))

        # These have a default behavior to return something meaningfull if 
        # no values are set
        self.train_ = self.data_.get_training_data()
        self.train_.set_batch_size(self.batch_size)
        self.validation_ = self.data_.get_validation_data()

        if isinstance(self.functions, list):
            self.functions_ = {k:1.0 for k in self.functions}
        else:
            self.functions_ = self.functions

        self.search_space_ = _brush.SearchSpace(self.train_, self.functions_)
        self.toolbox_ = self._setup_toolbox(data_train=self.train_, data_validation=self.validation_)

        self.archive_, self.logbook_ = nsga2(
            self.toolbox_, self.max_gen, self.pop_size, self.cx_prob, 
            (0.0<self.batch_size<1.0), self.verbosity, _brush.rnd_flt)


        final_ind_idx = 0

        # Each individual is a point in the Multi-Objective space. We multiply
        # the fitness by the weights so greater numbers are always better
        points = np.array([self.toolbox_.evaluateValidation(ind) for ind in self.archive_])
        points = points*np.array(self.weights)

        if self.validation_size==0.0:  # Using the multi-criteria decision making on training data
            # Selecting the best estimator using training data
            # (train data==val data if validation_size is set to 0.0)
            # and multi-criteria decision making

            # Normalizing
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            points = (points - min_vals) / (max_vals - min_vals)
            
            # Reference should be best value each obj. can have (after normalization)
            reference = np.array([1, 1])

            # closest to the reference
            final_ind_idx = np.argmin( np.linalg.norm(points - reference, axis=1) )
        else: # Best in obj.1 (loss) in validation data
            final_ind_idx = np.argmax( points[:, 0] )

        self.best_estimator_ = self.archive_[final_ind_idx].prg

        if self.verbosity > 0:
            print(f'best model {self.best_estimator_.get_model()}'+
                  f' with size {self.best_estimator_.size()}, '   +
                  f' depth {self.best_estimator_.depth()}, '      +
                  f' and fitness {self.archive_[0].fitness}'      )

        return self
    
    def _make_data(self, X, y=None, validation_size=0.0):
        # This function should not partition data (as it is used in predict).
        # partitioning is done in fit().

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(X, pd.DataFrame):
            # self.data_ = _brush.Dataset(X.to_dict(orient='list'), y)
            feature_names = X.columns.to_list()
            X = X.values
            if isinstance(y, NoneType):
                return _brush.Dataset(X,
                    feature_names=feature_names, validation_size=validation_size)
            else:
                return _brush.Dataset(X, y,
                    feature_names=feature_names, validation_size=validation_size)

        assert isinstance(X, np.ndarray)

        # if there is no label, don't include it in library call to Dataset
        if isinstance(y, NoneType):
            return _brush.Dataset(X, validation_size=validation_size)

        return _brush.Dataset(X, y, validation_size=validation_size)


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

        # Weight of each objective (+ for maximization, - for minimization)
        self.weights = (+1.0,-1.0)

    def _fitness_validation(self, ind, data: _brush.Dataset):
        # Fitness without fitting the expression, used with validation data
        return ( # (accuracy, size)
            (data.y==ind.prg.predict(data)).sum() / data.y.shape[0], 
            ind.prg.size()
        )

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
        
        if self.initialization not in ["grow", "full"]:
            raise ValueError(f"Invalid argument value for `initialization`. "
                             f"expected 'full' or 'grow'. got {self.initialization}")

        return creator.Individual(
            self.search_space_.make_classifier(
                self.max_depth,(0 if self.initialization=='grow' else self.max_size))
        if self.n_classes_ == 2 else
            self.search_space_.make_multiclass_classifier(
                self.max_depth, (0 if self.initialization=='grow' else self.max_size))
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

        # Weight of each objective (+ for maximization, - for minimization)
        self.weights = (-1.0,-1.0)

    def _fitness_validation(self, ind, data: _brush.Dataset):
        # Fitness without fitting the expression, used with validation data

        MSE = np.mean( (data.y-ind.prg.predict(data))**2 )
        if not np.isfinite(MSE): # numeric erros, np.nan, +-np.inf
            MSE = np.inf

        return ( MSE, ind.prg.size() )

    def _fitness_function(self, ind, data: _brush.Dataset):
        ind.prg.fit(data)

        MSE = np.mean( (data.y-ind.prg.predict(data))**2 )
        if not np.isfinite(MSE): # numeric erros, np.nan, +-np.inf
            MSE = np.inf

        return ( MSE, ind.prg.size() )

    def _make_individual(self):
        if self.initialization not in ["grow", "full"]:
            raise ValueError(f"Invalid argument value for `initialization`. "
                             f"expected 'full' or 'grow'. got {self.initialization}")
        
        return creator.Individual( # No arguments (or zero): brush will use PARAMS passed in set_params. max_size is sampled between 1 and params['max_size'] if zero is provided
            self.search_space_.make_regressor(
                self.max_depth, (0 if self.initialization=='grow' else self.max_size))
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