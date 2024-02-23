#include "module.h"
#include "bind_population.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

// using Reg = br::Program<ArrayXf>;
// using Cls = br::Program<ArrayXb>;
// using Rep = br::Program<ArrayXXf>;
// using MCls = br::Program<ArrayXi>;

void bind_populations(py::module& m)
{
    // TODO: make them a single class
    bind_population<br::ProgramType::Regressor>(m, "RegressorPopulation");
    bind_population<br::ProgramType::BinaryClassifier>(m, "ClassifierPopulation");
    
    bind_population<br::ProgramType::MulticlassClassifier>(m, "MultiClassifierPopulation");
    // bind_population<br::ProgramType::Representer>(m, "RepresenterPopulation");
}