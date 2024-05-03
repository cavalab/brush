#include "module.h"
#include "../vary/variation.h"
#include "../vary/variation.cpp" // TODO: figure out why im having symbol errors (if i dont include the cpp here as well)

#include "../pop/population.cpp"
#include "../pop/population.h"

namespace py = pybind11;
namespace nl = nlohmann;
namespace br = Brush;

template<br::ProgramType PT>
void bind_variation(py::module& m, string name)
{
    using Class = br::Var::Variation<PT>;

    // TODO: make variation a non-templated class
    py::class_<Class> vary(m, name.data() );

    vary.def(py::init<>([](br::Parameters& p, br::SearchSpace& ss){
             Class variation(p, ss);
             return variation; }))
        .def("mutate", &Class::mutate, py::return_value_policy::automatic)
        .def("cross", &Class::cross, py::return_value_policy::automatic)
        .def("vary_pop", [](Class &self, std::vector<br::Pop::Individual<PT>>& individuals, const Parameters& params) {

            if (individuals.size() != params.pop_size) {
                throw std::runtime_error("Individual vector has different number of individuals than pop_size. When calling variation, they should be the same. popsize is "+to_string(params.pop_size)+", number of individuals is " + to_string(individuals.size()));
            }

            auto pop = br::Pop::Population<PT>();

            pop.init(individuals, params);
        
            vector<br::Pop::Individual<PT>> pool;
            pool.resize(0);

            for (int island = 0; island < params.num_islands; ++island)
            {
                // I am assuming the individual vector passed as argument will contain the selected parents already
                vector<size_t> parents = pop.get_island_indexes(island);

                // including offspring indexes (the vary method will store the offspring in the second half of the index vector)
                pop.add_offspring_indexes(island);
                
                self.vary(pop, island, parents, params);

                // making copies of the second half of the island individuals
                vector<size_t> idxs = pop.get_island_indexes(island);
                int start = idxs.size()/2;
                for (unsigned i = start; i<idxs.size(); ++i)
                {
                    // this is where the offspring is saved
                    pool.push_back(pop[idxs.at(i)]);
                }
            }

            // returns references   
            return pool;
        })
        ;
}