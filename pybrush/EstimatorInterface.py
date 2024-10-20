"""
Estimator interface for GP implementations.

This interface defines all the hyperparameters for Brush estimators and
provides documentation for the hyperparameters.
"""

import numpy as np
from pybrush import Parameters

class EstimatorInterface():
    """
    Interface class for all estimators in pybrush.

    Parameters
    ----------
    mode : str, default 'classification'
        The mode of the estimator. Used by subclasses
    pop_size : int, default 100
        Population size.
    max_gens : int, default 100
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
        This also corresponds to the number of parallel threads in the c++
        engine.
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
        This is used only in DeapEstimators.
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
    use_arch: boolean, optional (default: False)
        Determines if we should save pareto front of the entire evolution
        (when set to  True) or just the final population (False).
    batch_size : float, default 1.0
        Percentage of training data to sample every generation. If `1.0`, then
        all data is used. Very small values can improve execution time, but 
        also lead to underfit.
    save_population: str, optional (default "")
        string containing the path to save the final population. Ignored if
        not provided.
    load_population: str, optional (default "")
        string containing the path to load the initial population. Ignored
        if not provided.
    shuffle_split: boolean, optional (default False)
        whether if the engine should shuffle the data before splitting it
        into train and validation partitions. Ignored if `validation_size`
        is set to zero.
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
    """

    def __init__(self,
        mode='classification',
        pop_size=100,
        max_gens=100,
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
        save_population="",
        load_population="",
        shuffle_split=False,
        weights_init=True,
        val_from_arch=True,
        use_arch=False,
        validation_size: float = 0.0,
        batch_size: float = 1.0
    ):
        self.pop_size=pop_size
        self.max_gens=max_gens
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
        self.save_population=save_population
        self.load_population=load_population
        self.mutation_probs=mutation_probs
        self.val_from_arch=val_from_arch # TODO: val from arch implementation (in cpp side)
        self.use_arch=use_arch
        self.functions=functions
        self.objectives=objectives
        self.shuffle_split=shuffle_split
        self.initialization=initialization
        self.random_state=random_state
        self.batch_size=batch_size
        self.weights_init=weights_init
        self.validation_size=validation_size

    def _wrap_parameters(self, **extra_kwargs):
        """
        Creates a `Parameters` class to send to c++ backend the settings for
        the algorithm to use.
        """
        
        if isinstance(self.functions, list):
            self.functions_ = {k:1.0 for k in self.functions}
        else:
            self.functions_ = self.functions

        params = Parameters()

        params.classification = self.mode == "classification"
        params.n_classes = self.n_classes_
        params.verbosity = self.verbosity
        params.n_jobs = self.n_jobs
        params.pop_size = self.pop_size
        params.max_gens = self.max_gens
        params.logfile = self.logfile
        params.save_population = self.save_population
        params.load_population = self.load_population
        params.max_stall = self.max_stall
        params.max_time = self.max_time
        params.num_islands = self.num_islands
        params.max_depth = self.max_depth
        params.max_size = self.max_size
        params.objectives = self.objectives
        params.shuffle_split = self.shuffle_split
        params.cx_prob = self.cx_prob
        params.use_arch = self.use_arch
        params.val_from_arch = self.val_from_arch
        params.mig_prob = self.mig_prob
        params.functions = self.functions_
        params.mutation_probs = self.mutation_probs
        params.validation_size = self.validation_size
        params.batch_size = self.batch_size
        params.feature_names = self.feature_names_
    
        params.scorer_ = "mse"
        if self.mode == "classification":
            params.scorer_ = "log" if self.n_classes_ == 2 else "multi_log"

        if self.random_state is not None:
            seed = 0
            if isinstance(self.random_state, np.random.Generator):
                seed = self.random_state.integers(1_000_000)
            elif isinstance(self.random_state, int):
                seed = self.random_state
            else:
                raise ValueError("random_state must be either a numpy random generator or an integer")

            params.random_state = seed

        for k, v in extra_kwargs.items():
            setattr(params, k, v)

        return params