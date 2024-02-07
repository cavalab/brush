#include "module.h"
#include "bind_variation.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

// using Reg = br::Program<ArrayXf>;
// using Cls = br::Program<ArrayXb>;
// using Rep = br::Program<ArrayXXf>;
// using MCls = br::Program<ArrayXi>;

void bind_variations(py::module& m)
{
    bind_variation<br::ProgramType::Regressor>(m, "RegressorVariator");
    bind_variation<br::ProgramType::BinaryClassifier>(m, "ClassifierVariator");
    
    bind_variation<br::ProgramType::MulticlassClassifier>(m, "MultiClassifierVariator");
    bind_variation<br::ProgramType::Representer>(m, "RepresenterVariator");
}