#include "module.h"
#include "bind_evaluator.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

void bind_evaluators(py::module &m)
{
    bind_evaluator<br::ProgramType::Regressor>(m, "RegressorEvaluator");
    bind_evaluator<br::ProgramType::BinaryClassifier>(m, "ClassifierEvaluator");
    bind_evaluator<br::ProgramType::MulticlassClassifier>(m, "MultiClassifierEvaluator");
    bind_evaluator<br::ProgramType::Representer>(m, "RepresenterEvaluator");
}