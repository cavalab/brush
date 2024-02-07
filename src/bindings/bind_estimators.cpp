#include "module.h"
#include "bind_estimators.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

void bind_estimators(py::module& m)
{
     bind_estimator<Reg>(m, "BrushRegressorEstimator");
     bind_estimator<Cls>(m, "BrushClassifierEstimator");
     
     // TODO: make these work
     bind_estimator<br::MulticlassClassifierEstimator>(m, "BrushMultiClassifierEstimator");
     bind_estimator<br::RepresenterEstimator>(m, "BrushRepresenterEstimator");
}