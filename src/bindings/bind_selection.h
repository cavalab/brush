#include "module.h"
// TODO: figure out why im having symbol errors (if i dont include the cpp here as well)
#include "../selection/selection.h"
#include "../selection/selection.cpp"
#include "../selection/selection_operator.h"
#include "../selection/selection_operator.cpp"
#include "../selection/nsga2.h"
#include "../selection/nsga2.cpp"

#include "../population.cpp"
#include "../population.h"

// #include "../individual.h"
//#include "../selection/selection.cpp" 

namespace py = pybind11;
namespace nl = nlohmann;
namespace br = Brush;

template<br::ProgramType PT>
void bind_selection(py::module& m, string name)
{
    using Class = br::Sel::Selection<PT>;

    // TODO: make selection a non-templated class
    py::class_<Class> sel(m, name.data() );

    sel.def(py::init<>())
       .def(py::init(
           [](string type, bool survival){ Class s(type, survival); return s; })
       )
       .def("select", [](Class &self, std::vector<br::Pop::Individual<PT>>& individuals,
                         const Parameters& params) {
                            
            // auto sel = Class("nsga2", false);
            auto pop = br::Pop::Population<PT>();

            pop.init(individuals, params);

            vector<size_t> selected = self.select(pop, 0, params);

            vector<br::Pop::Individual<PT>> pool;
            pool.resize(0);

            for (size_t idx : selected) {
                pool.push_back(pop[idx]);
            }

            // returns references   
            return pool;
       })
       .def("survive", [](Class &self, std::vector<br::Pop::Individual<PT>>& individuals,
                         const Parameters& params) {
                            
            // auto sel = Class("nsga2", false);
            auto pop = br::Pop::Population<PT>();

            // std::cout << "created new population" << std::endl;

            pop.init(individuals, params);

            // std::cout << "called init with individuals" << std::endl;

            vector<size_t> selected = self.survive(pop, 0, params);

            // std::cout << "survival" << std::endl;

            vector<br::Pop::Individual<PT>> pool;

            // std::cout << "starting to fill the pool" << std::endl;

            pool.resize(0);

            // std::cout << "pool is empty" << std::endl;

            for (size_t idx : selected) {
                pool.push_back(pop[idx]);
            }

            // std::cout << "pool has size" << pool.size() << std::endl;

            // returns references   
            return pool;
       })
       ;
}