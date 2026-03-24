from ._versionstr import __version__

# Interfaces for Brush data structures. Use to prototype with Brush
from ._brush import Dataset
from ._brush import SearchSpace
from ._brush import Parameters

# geting random floats with brush (avoid random state issues in parallel exec)
from ._brush import rnd_flt as brush_rng

from ._brush import individual # Individual classes (specific for each task)

# c++ learning engines
from ._brush.engine import *

# Evaluation, selection, and variation. used in python estimators
from ._brush import RegressorEvaluator, ClassifierEvaluator, MultiClassifierEvaluator
from ._brush import RegressorSelector, ClassifierSelector, MultiClassifierSelector
from ._brush import RegressorVariator, ClassifierVariator, MultiClassifierVariator

from pybrush.BrushEstimator import BrushClassifier, BrushRegressor

# deap api
try:
    from pybrush import deap_api 
except ImportError:
    import warnings
    
    class _DeapAPIWarning:
        def __getattr__(self, name):
            warnings.warn(
                "deap_api could not be imported. Please install required dependencies.",
                ImportWarning,
                stacklevel=2
            )
            raise AttributeError(f"deap_api is not available")
    
    deap_api = _DeapAPIWarning()
