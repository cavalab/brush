#include "module.h"
#include "bind_engines.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

void bind_engines(py::module& m)
{
     bind_engine<Reg>(m, "RegressorEngine");
     bind_engine<Cls>(m, "ClassifierEngine");
     
     // TODO: make these work
     bind_engine<br::MulticlassClassifierEngine>(m, "MultiClassifierEngine");
     bind_engine<br::RepresenterEngine>(m, "RepresenterEngine");
}