#include "module.h"
#include "bind_cbrush.h" // TODO: rename it to bind_estimators

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

void bind_estimators(py::module& m)
{
     bind_estimator<br::RegressorEstimator>(m, "Regressor");
     bind_estimator<br::ClassifierEstimator>(m, "Classifier");
     bind_estimator<br::MulticlassClassifierEstimator>(m, "MultiClassifier");
     bind_estimator<br::RepresenterEstimator>(m, "Representer");
}