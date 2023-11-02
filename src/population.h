#ifndef POPULATION_H
#define POPULATION_H

#include "program/program.h"
#include "search_space.h"

using std::vector;
using std::string;
using Eigen::Map;

namespace Brush
{   
namespace Pop{

////////////////////////////////////////////////////////////////// Declarations
extern int last;
/*!
 * @class Population
 * @brief Defines a population of programs and functions for constructing them. 
 */
template<ProgramType T> 
struct Population
{
    vector<Program<T>*> individuals;  ///< individual programs

    Population(int p = 0);
    
    ~Population();
    
    /// initialize population of programs with a starting model and/or from file 
    void init(const Program<T>& starting_model, 
              const Parameters& params, 
              const SearchSpace& ss
              );
    
    /// update individual vector size 
    void resize(int pop_size);
    
    /// reduce programs to the indices in survivors. 
    void update(vector<size_t> survivors);
    
    /// returns population size
    int size();

    /// adds a program to the population. 

    void add(Program<T>&);
    
    /// setting and getting from individuals vector
    const Program<T> operator [](size_t i) const;
    const Program<T>& operator [](size_t i);

    /// return population equations. 
    string print_eqns(bool just_offspring=false, string sep="\n");

    /// return complexity-sorted Pareto front indices. 
    vector<size_t> sorted_front(unsigned);
    
    /// Sort population in increasing complexity.
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
        Population & pop;
        SameFitComplexity(Population& p): pop(p){}
        bool operator()(size_t i, size_t j)
        {
            return (pop.individuals[i].fitness == pop.individuals[j].fitness &&
                   pop.individuals[i].set_complexity() == pop.individuals[j].set_complexity());
        }
    };

    // save serialized population
    void save(string filename);
    // load serialized population
    void load(string filename);
};        

// //TODO
// /* void from_json(const json& j, Population& p); */
// /* void to_json(json& j, const Population& p); */
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Population, individuals);
}//Pop

}//FT    
#endif
