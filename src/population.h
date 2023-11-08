#ifndef POPULATION_H
#define POPULATION_H

#include "search_space.h"
#include "individual.h"
#include "program/program.h"

using std::vector;
using std::string;
using Eigen::Map;

namespace Brush {   
namespace Pop {

template<ProgramType T> 
class Population{
private:
    void set_island_ranges();
public:
    bool offspring_ready;
    vector<Individual<T>*> individuals;
    vector<tuple<size_t, size_t>> island_ranges;
    vector<size_t> island_skip; // number of indexes to skip for each island (when variation fails)
    unsigned int n_islands;
    float mig_prob;

    Population(int p = 0, int n_islands=1);
    
    ~Population();
    
    /// initialize population of programs with a starting model and/or from file 
    void init(const SearchSpace& ss, const Parameters& params);

    /// returns population size
    int size() { return individuals.size(); };

    tuple<size_t, size_t> get_island_range(int island) {
        return island_ranges.at(island); };

    /// update individual vector size, distributing the expressions in n_islands
    void prep_offspring_slots();
    
    // TODO: WORK WITH ISLANDS
    /// reduce programs to the indices in survivors. 
    void update(vector<size_t> survivors);
    
    /// setting and getting from individuals vector (will ignore islands)
    const Individual<T> operator [](size_t i) const {return individuals.at(i);}
    const Individual<T> & operator [](size_t i) {return individuals.at(i);}

    /// return population equations. 
    string print_models(bool just_offspring=false, string sep="\n");

    /// return complexity-sorted Pareto front indices for each island
    vector<vector<size_t>> sorted_front(unsigned rank=1);

    // pareto front ignoring island divisions
    vector<size_t> hall_of_fame(unsigned rank=1);
    
    // perform a migration in the population. Individuals from sorted front or hall of fame will replace others by the
    // probability set in parameters. Expects a population without offspring
    void migrate();

    /// Sort each island in increasing complexity.
    struct SortComplexity
    {
        Population& pop;
        SortComplexity(Population& p): pop(p){}
        bool operator()(size_t i, size_t j)
        { 
            return pop.individuals[i].set_complexity() < pop.individuals[j].set_complexity();
        }
    };
    
    /// check for same fitness and complexity to filter uniqueness. 
    struct SameFitComplexity
    {
        Population<T> & pop;
        SameFitComplexity(Population<T>& p): pop(p){}
        bool operator()(size_t i, size_t j)
        {
            return (pop.individuals[i].fitness == pop.individuals[j].fitness &&
                   pop.individuals[i].set_complexity() == pop.individuals[j].set_complexity());
        }
    };
};

}// Pop
}// Brush

#endif
