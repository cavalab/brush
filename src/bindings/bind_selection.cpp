#include "module.h"
#include "bind_selection.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

// using Reg = br::Program<ArrayXf>;
// using Cls = br::Program<ArrayXb>;
// using Rep = br::Program<ArrayXXf>;
// using MCls = br::Program<ArrayXi>;

void bind_selections(py::module& m)
{
    // TODO: make them a single class
    bind_selection<br::ProgramType::Regressor>(m, "RegressorSelector");
    bind_selection<br::ProgramType::BinaryClassifier>(m, "ClassifierSelector");
    
    bind_selection<br::ProgramType::MulticlassClassifier>(m, "MultiClassifierSelector");
    // bind_selection<br::ProgramType::Representer>(m, "RepresenterSelector");
}