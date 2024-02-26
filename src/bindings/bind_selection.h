#include "module.h"
// TODO: figure out why im having symbol errors (if i dont include the cpp here as well)
#include "../selection/selection.h"
#include "../selection/selection.cpp"
#include "../selection/selection_operator.h"
#include "../selection/selection_operator.cpp"
#include "../selection/nsga2.h"
#include "../selection/nsga2.cpp"
#include "../selection/lexicase.h"
#include "../selection/lexicase.cpp"

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

            vector<br::Pop::Individual<PT>> pool;
            pool.resize(0);

            for (int island = 0; island < params.num_islands; ++island)
            {
                vector<size_t> selected = self.select(pop, island, params);

                // std::cout << "selecting in island " << island << std::endl;

                for (size_t idx : selected) {
                    pool.push_back(pop[idx]);
                }
            }

            // returns references   
            return pool;
       })
       .def("survive", [](Class &self, std::vector<br::Pop::Individual<PT>>& individuals,
                         const Parameters& params) {
                            
            // auto sel = Class("nsga2", false);
            auto pop = br::Pop::Population<PT>();

            pop.init(individuals, params);

            vector<br::Pop::Individual<PT>> pool;
            pool.resize(0);

            for (int island = 0; island < params.num_islands; ++island)
            {
                vector<size_t> selected = self.survive(pop, island, params);

                for (size_t idx : selected) {
                    pool.push_back(pop[idx]);
                }
            }

            // returns references   
            return pool;
       })
       .def("migrate", [](Class &self, std::vector<br::Pop::Individual<PT>>& individuals,
                         const Parameters& params) {

            auto pop = br::Pop::Population<PT>();

            pop.init(individuals, params);
            pop.migrate(); // this will modify island indexes inplace

            vector<br::Pop::Individual<PT>> pool;
            pool.resize(0);

            for (int island = 0; island < params.num_islands; ++island)
            {
                vector<size_t> selected = pop.get_island_indexes(island);

                for (size_t idx : selected) {
                    pool.push_back(pop[idx]);
                }
            }
            // returns references   
            return pool;
       })
       ;
}