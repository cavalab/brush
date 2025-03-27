#include "module.h"
#include "bind_individuals.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;


void bind_individuals(py::module& m)
{
     bind_individual<br::ProgramType::Regressor>(m, "RegressorIndividual");
     bind_individual<br::ProgramType::BinaryClassifier>(m, "ClassifierIndividual");
     bind_individual<br::ProgramType::MulticlassClassifier>(m, "MultiClassifierIndividual");
     bind_individual<br::ProgramType::Representer>(m, "RepresenterIndividual");
}