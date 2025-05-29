#include "module.h"

#include "../pop/population.h"
#include "../pop/population.cpp"

#include "../bandit/bandit.h"
#include "../bandit/bandit_operator.h"
#include "../bandit/dummy.h"
#include "../bandit/thompson.h"
#include "../ind/individual.h"
#include "../ind/individual.cpp"

#include "../simplification/constants.cpp"
#include "../simplification/constants.h"
#include "../simplification/inexact.cpp"
#include "../simplification/inexact.h"

#include "../vary/variation.h"
#include "../vary/variation.cpp"

namespace py = pybind11;
namespace nl = nlohmann;
namespace br = Brush;

template<br::ProgramType PT>
void bind_variation(py::module& m, string name)
{
    using Class = br::Var::Variation<PT>;

    // TODO: make variation a non-templated class
    py::class_<Class> vary(m, name.data() );

    vary.def(py::init<>([](br::Parameters& p, br::SearchSpace& ss, br::Data::Dataset& d){
             Class variation(p, ss, d);
             return variation; }))
        .def("mutate", &Class::mutate, py::return_value_policy::automatic)
        .def("cross", &Class::cross, py::return_value_policy::automatic)
        .def("vary_pop", [](Class &self, 
                            std::vector<br::Pop::Individual<PT>>& individuals,
                            const Parameters& params) {
            if (individuals.size() != params.pop_size) {
                string msg = "Individual vector has different number of "
                             "individuals than pop_size. When calling "
                             "variation, they should be the same. popsize is "+
                             to_string(params.pop_size)+", number of "
                             "individuals is "+to_string(individuals.size());

                throw std::runtime_error(msg);
            }

            auto pop = br::Pop::Population<PT>();

            pop.init(individuals, params);
        
            vector<br::Pop::Individual<PT>> pool;
            pool.resize(0);

            for (int island = 0; island < params.num_islands; ++island)
            {
                // I am assuming the individual vector passed as argument
                // will contain the selected parents already
                vector<size_t> parents = pop.get_island_indexes(island);

                // including offspring indexes (the vary method will store the
                // offspring in the second half of the index vector)
                pop.add_offspring_indexes(island);
                
                self.vary(pop, island, parents);

                // making copies of the second half of the island individuals
                vector<size_t> indices = pop.get_island_indexes(island);
                int start = indices.size()/2;
                for (unsigned i = start; i<indices.size(); ++i)
                {
                    // this is where the offspring is saved
                    pool.push_back(pop[indices.at(i)]);
                }
            } 
            return pool;
        })
        ;
}
