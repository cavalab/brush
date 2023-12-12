#include "module.h"
#include "bind_individuals.h"

namespace py = pybind11;
namespace br = Brush;
namespace nl = nlohmann;

void bind_individuals(py::module& m)
{
     bind_individual<br::RegressorIndividual>(m, "Regressor");
     bind_individual<br::ClassifierIndividual>(m, "Classifier");
     bind_individual<br::MulticlassClassifierIndividual>(m, "MultiClassifier");
     bind_individual<br::RepresenterIndividual>(m, "Representer");
}