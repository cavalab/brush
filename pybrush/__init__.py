# Interfaces for Brush classes. Use to prototype with Brush
from _brush import Dataset, SearchSpace, Parameters # TODO: make individual wrapper, Individual

# Brush's original EA algorithm
from pybrush.BrushEstimator import BrushClassifier, BrushRegressor

# Prototyping an EA using brush classes, but other EA framework
from pybrush.DeapEstimator import DeapClassifier, DeapRegressor