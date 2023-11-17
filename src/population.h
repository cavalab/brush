#ifndef POPULATION_H
#define POPULATION_H

#include "search_space.h"
#include "individual.h"
#include "program/program.h"
#include "util/error.h"

using std::vector;
using std::string;
using Eigen::Map;

namespace Brush {   
namespace Pop {

template<ProgramType T> 
class Population{
public:
    size_t pop_size;
    unsigned int n_islands;
    float mig_prob;

    vector<std::shared_ptr<Individual<T>>> individuals;

    // TODO: MAKE SURE THIS TWO ITEMS BELOW ARE TAKEN CARE IN THE MAIN LOOP AND IN TEST_POPULATION (I may need to create new methods for taking care of this)
    // - fitting, fitness calculation, and setting the objectives are not thread safe because we write in individual attributes.
    // - prepare offspring and update are not thread safe because we insert/delete elements from the array. 
    vector<vector<size_t>> island_indexes;

    // TODO: taskflow needs to use n_islands as n_jobs
    Population();
    
    ~Population(){};
    
    /// initialize population of programs with a starting model and/or from file 
    void init(SearchSpace& ss, const Parameters& params);

    /// returns population size (the effective size of the individuals)
    int size() { return individuals.size(); };

    vector<size_t> get_island_indexes(int island){ return island_indexes.at(island); };

    /// update individual vector size, distributing the expressions in n_islands
    // TODO: add elements to the end
    void prep_offspring_slots(int island);
    
    /// reduce programs to the indices in survivors. Not thread safe,as it removes elements
    void update(vector<vector<size_t>> survivors);
    
    /// setting and getting from individuals vector (will ignore islands)
    const Individual<T>& operator [](size_t i) const {return *individuals.at(i);}
    const Individual<T>& operator [](size_t i) {return *individuals.at(i);}

    /// return population equations. 
    string print_models(bool just_offspring=false, string sep="\n");

    /// return complexity-sorted Pareto front indices for each island
    vector<vector<size_t>> sorted_front(unsigned rank=1);

    // pareto front ignoring island divisions
    vector<size_t> hall_of_fame(unsigned rank=1);
    
    // perform a migration in the population. Individuals from sorted front or hall of fame will replace others by the
    // probability set in parameters. Expects a population without offspring
    void migrate(); // TODO: change just the indexes

    /// Sort each island in increasing complexity. This is not thread safe. I should set complexities of the whole population before calling it, and use get_complexity instead
    struct SortComplexity
    {
        Population& pop;
        SortComplexity(Population& p): pop(p){}
        bool operator()(size_t i, size_t j)
        { 
            return pop[i].get_complexity() < pop[j].get_complexity();
        }
    };
    
    /// check for same fitness and complexity to filter uniqueness. 
    struct SameFitComplexity
    {
        Population<T> & pop;
        SameFitComplexity(Population<T>& p): pop(p){}
        bool operator()(size_t i, size_t j)
        {
            return (pop[i].fitness == pop[j].fitness
                   && pop[i].get_complexity() == pop[j].get_complexity());
        }
    };
};

}// Pop
}// Brush

#endif
