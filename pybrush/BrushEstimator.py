"""
sklearn-compatible wrapper for GP analyses.

TODO: update this docstring
See brushgp.cpp for Python (via pybind11) modules that give more fine-grained
control of the underlying GP objects.
"""
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.validation  import check_is_fitted
# from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from _brush.individual import * # RegressorIndividual, ClassifierIndividual, MultiClassifierIndividual
from _brush.engine import * # Regressor, Classifier, and MultiClassifier engines
from pybrush import Parameters, Dataset, SearchSpace
from pybrush import brush_rng


class BrushEstimator(BaseEstimator):
    """
    This is the base class for Deap-based Brush estimators. 
    This class shouldn't be called directly; instead, call a child class like 
    :py:class:`DeapRegressor <brush.estimator.DeapRegressor>` or :py:class:`DeapClassifier <brush.estimator.DeapClassifier>`. 
    All of the shared parameters are documented here. 

    Parameters
    ----------
    mode : str, default 'classification'
        The mode of the estimator. Used by subclasses
    pop_size : int, default 100
        Population size.
    gens : int, default 100
        Maximum iterations of the algorithm.
    max_time: int, optional (default: -1)
        Maximum time terminational criterion in seconds. If -1, not used.
    max_stall: int, optional (default: 0)
        How many generations to continue after the validation loss has
        stalled. If 0, not used.
    verbosity : int, default 0
        Controls level of printouts.
    max_depth : int, default 0
        Maximum depth of GP trees in the GP program. Use 0 for no limit.
    max_size : int, default 0
        Maximum number of nodes in a tree. Use 0 for no limit.
    num_islands : int, default 5
        Number of independent islands to use in evolutionary framework. 
        Ignored if `algorithm!="nsga2island"`.
    mig_prob : float, default 0.05
        Probability of occuring a migration between two random islands at the
        end of a generation, must be between 0 and 1.
    cx_prob : float, default 1/7
        Probability of applying the crossover variation when generating the offspring,
        must be between 0 and 1.
        Given that there are `n` mutations, and either crossover or mutation is 
        used to generate each individual in the offspring (but not both at the
        same time), we want to have by default an uniform probability between
        crossover and every possible mutation. By setting `cx_prob=1/(n+1)`, and
        `1/n` for each mutation, we can achieve an uniform distribution.
    mutation_probs : dict, default {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6, "toggle_weight_on":1/6, "toggle_weight_off":1/6}
        A dictionary with keys naming the types of mutation and floating point 
        values specifying the fraction of total mutations to do with that method.
        The probability of having a mutation is `(1-cx_prob)` and, in case the mutation
        is applied, then each mutation option is sampled based on the probabilities
        defined in `mutation_probs`. The set of probabilities should add up to 1.0.
    functions: dict[str,float] or list[str], default {}
        A dictionary with keys naming the function set and values giving the probability
        of sampling them, or a list of functions which will be weighted uniformly.
        If empty, all available functions are included in the search space.
    initialization : {"uniform", "max_size"}, default "uniform" 
        Distribution of sizes on the initial population. If `max_size`, then every
        expression is created with `max_size` nodes. If `uniform`, size will be
        uniformly distributed between 1 and `max_size`.
    objectives : list[str], default ["error", "size"]
        list with one or more objectives to use. Options are `"error", "size", "complexity"`.
        If `"error"` is used, then it will be the mean squared error for regression,
        and accuracy for classification.
    algorithm : {"nsga2island", "nsga2", "gaisland", "ga"}, default "nsga2"
        Which Evolutionary Algorithm framework to use to evolve the population.
    weights_init : bool, default True
        Whether the search space should initialize the sampling weights of terminal nodes
        based on the correlation with the output y. If `False`, then all terminal nodes
        will have the same probability of 1.0.
    validation_size : float, default 0.0
        Percentage of samples to use as a hold-out partition. These samples are used
        to calculate statistics during evolution, but not used to train the models.
        The `best_estimator_` will be selected using this partition. If zero, then
        the same data used for training is used for validation.
    val_from_arch: boolean, optional (default: True)
        Validates the final model using the archive rather than the whole 
        population.
    batch_size : float, default 1.0
        Percentage of training data to sample every generation. If `1.0`, then
        all data is used. Very small values can improve execution time, but 
        also lead to underfit.
    logfile: str, optional (default: "")
        If specified, spits statistics into a logfile. "" means don't log.
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
    best_estimator_ : pybrush.Program
        The final model picked from training. Used in subsequent calls to :func:`predict`. 
    archive_ : list[deap_api.DeapIndividual]
        The final population from training. 
    data_ : pybrush.Dataset
        The complete data in Brush format. 
    train_ : pybrush.Dataset
        Partition of `data_` containing `(1-validation_size)`% of the data, in Brush format.
    validation_ : pybrush.Dataset
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
        gens=100,
        max_time=-1,
        max_stall=0,
        verbosity=0,
        max_depth=3,
        max_size=20,
        num_islands=1,
        n_jobs=1,
        mig_prob=0.05,
        cx_prob= 1/7,
        mutation_probs = {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6,
                          "toggle_weight_on":1/6, "toggle_weight_off":1/6},
        functions: list[str]|dict[str,float] = {},
        initialization="uniform",
        algorithm="nsga2",
        objectives=["error", "size"],
        random_state=None,
        logfile="",
        weights_init=True,
        val_from_arch=True,
        validation_size: float = 0.0,
        batch_size: float = 1.0
        ):

        self.pop_size=pop_size
        self.gens=gens
        self.max_stall=max_stall
        self.max_time=max_time
        self.verbosity=verbosity
        self.algorithm=algorithm
        self.mode=mode
        self.max_depth=max_depth
        self.max_size=max_size
        self.num_islands=num_islands
        self.mig_prob=mig_prob
        self.n_jobs=n_jobs
        self.cx_prob=cx_prob
        self.logfile=logfile
        self.mutation_probs=mutation_probs
        self.val_from_arch=val_from_arch # TODO: val from arch
        self.functions=functions
        self.objectives=objectives
        self.initialization=initialization
        self.random_state=random_state
        self.batch_size=batch_size
        self.weights_init=weights_init
        self.validation_size=validation_size

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
        
        self.feature_names_ = []
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.to_list()

        self.data_ = self._make_data(X, y, 
                                     feature_names=self.feature_names_,
                                     validation_size=self.validation_size)

        if isinstance(self.functions, list):
            self.functions_ = {k:1.0 for k in self.functions}
        else:
            self.functions_ = self.functions

        # set n classes if relevant
        self.n_classes_ = 0
        if self.mode=="classification":
            self.n_classes_ = len(np.unique(y))

        # These have a default behavior to return something meaningfull if 
        # no values are set
        self.train_ = self.data_.get_training_data()
        self.train_.set_batch_size(self.batch_size) # TODO: update batch indexes at the beggining of every generation
        self.validation_ = self.data_.get_validation_data()

        self.search_space_ = SearchSpace(self.data_, self.functions_, self.weights_init)
                
        self.parameters_ = Parameters()
        self.parameters_.classification = self.mode == "classification"
        self.parameters_.n_classes = self.n_classes_
        self.parameters_.verbosity = self.verbosity
        self.parameters_.n_jobs = self.n_jobs
        self.parameters_.pop_size = self.pop_size
        self.parameters_.gens = self.gens
        self.parameters_.logfile = self.logfile
        self.parameters_.max_stall = self.max_stall
        self.parameters_.max_time = self.max_time
        self.parameters_.num_islands = self.num_islands
        self.parameters_.max_depth = self.max_depth
        self.parameters_.max_size = self.max_size
        self.parameters_.objectives = self.objectives
        self.parameters_.cx_prob = self.cx_prob
        self.parameters_.mig_prob = self.mig_prob
        self.parameters_.functions = self.functions
        self.parameters_.mutation_probs = self.mutation_probs
        self.parameters_.validation_size = self.validation_size
        self.parameters_.batch_size = self.batch_size
        self.parameters_.feature_names = self.feature_names_
    
        self.parameters_.scorer_ = "mse"
        if self.mode == "classification":
            self.parameters_.scorer_ = "log" if self.n_classes_ == 2 else "multi_log"

        if self.random_state is not None:
            seed = 0
            if isinstance(self.random_state, np.random.Generator):
                seed = self.random_state.integers(10000)
            elif isinstance(self.random_state, int):
                seed = self.random_state
            else:
                raise ValueError("random_state must be either a numpy random generator or an integer")

            self.parameters_.random_state = seed

        self.engine_ = None
        if self.mode == 'classification':
            self.engine_ = ( ClassifierEngine
                             if self.n_classes_ == 2 else
                             MultiClassifierEngine)(self.parameters_)
        else:
            self.engine_ = RegressorEngine(self.parameters_)

        self.engine_.fit(self.data_)
        self.best_estimator_ = self.engine_.best_ind

        return self
    
    def _make_data(self, X, y=None, feature_names=[], validation_size=0.0):
        # This function should not partition data (since it may be used in `predict`).
        # partitioning is done by `fit`. Feature names should be inferred
        # before calling _make_data (so predict can be made with np arrays or
        # pd dataframes).

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        assert isinstance(X, np.ndarray)

        if y is None:
            return Dataset(X=X,
                    feature_names=feature_names, c=self.mode == "classification", 
                    validation_size=validation_size)

        return Dataset(X=X, y=y,
            feature_names=feature_names, c=self.mode == "classification",
            validation_size=validation_size)


    def predict(self, X):
        """Predict using the best estimator in the archive. """

        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        data = Dataset(X=X, ref_dataset=self.data_, c=self.mode == "classification",
                       feature_names=self.feature_names_)
        
        # data = self._make_data(X, feature_names=self.feature_names_)

        return self.best_estimator_.program.predict(data)

    def get_params(self, deep=True):
        out = dict()
        for (key, value) in self.__dict__.items():
            if not key.endswith('_'):
                if deep and hasattr(value, "get_params") and not isinstance(value, type):
                    deep_items = value.get_params().items()
                    out.update((key + "__" + k, val) for k, val in deep_items)
                out[key] = value
        return out
    

class BrushClassifier(BrushEstimator,ClassifierMixin):
    """Deap-based Brush for classification.

    For options, see :py:class:`DeapEstimator <brush.estimator.DeapEstimator>`. 

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    >>> X = df.drop(columns='target')
    >>> y = df['target']
    >>> from pybrush import DeapClassifier
    >>> est = DeapClassifier()
    >>> est.fit(X,y)
    >>> # print('score:', est.score(X,y))
    """
    def __init__( self, **kwargs):
        super().__init__(mode='classification',**kwargs)

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
        
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        data = Dataset(X=X, ref_dataset=self.data_, c=True,
                              feature_names=self.feature_names_)

        # data = self._make_data(X, feature_names=self.feature_names_)

        prob = self.best_estimator_.program.predict_proba(data)

        if self.n_classes_ == 2:
            prob = np.hstack( (np.ones(X.shape[0]).reshape(-1,1), prob.reshape(-1,1)) )  
            prob[:, 0] -= prob[:, 1]

        return prob


class BrushRegressor(BrushEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(mode='regressor',**kwargs)