# Interfaces for Brush classes. Use to prototype with Brush
from _brush import Dataset
from _brush import SearchSpace
from _brush import Parameters

# Individuals
from _brush.individual import RegressorIndividual, \
                              ClassifierIndividual, MultiClassifierIndividual

# Prototyping an EA using brush classes, but other EA framework
from pybrush.DeapEstimator import DeapClassifier, DeapRegressor