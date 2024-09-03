#include "archive.h"

namespace Brush {
namespace Pop {

template<ProgramType T>
Archive<T>::Archive():  selector(true) {};

template<ProgramType T>
void Archive<T>::set_objectives(vector<string> objectives)
{
    this->sort_complexity = in(objectives, std::string("complexity"));
}

// sorting etc --- all done using fitness class (easier to compare regardless of obj func)
template<ProgramType T>
bool Archive<T>::sortComplexity(const Individual<T>& lhs, 
        const Individual<T>& rhs)
{
    // TODO: use getters for all info in fitness (instead of directly accessing them?).
    // other option would be having the getters and setters to use iin pybind11, but
    // in cpp we do it directly (we know how to manipulate this thing, but users may not,
    // so these setters could do some validation to justify its existence).

    return lhs.fitness.complexity < rhs.fitness.complexity;
}

template<ProgramType T>
bool Archive<T>::sortObj1(const Individual<T>& lhs, 
        const Individual<T>& rhs)
{
    // sort based on index (we can have more than 2 obj in brush implementation)
    // obs: because of the weights, every objective is a maximization problem
    // when comparing weighted values (which should be the right way of doing it)
    // the bigger the better. the weights allow us to use different min/max metrics
    // without having to deal with this particular details

    float lhs_obj1 = lhs.fitness.wvalues.at(0);
    float rhs_obj1 = rhs.fitness.wvalues.at(0);

    return lhs_obj1 > rhs_obj1;
}

template<ProgramType T>
bool Archive<T>::sameFitComplexity(const Individual<T>& lhs, 
        const Individual<T>& rhs)
{
    // fitness' operator== is overloaded to compare wvalues.
    // we also check complexity equality to avoid the case where the user
    // did not specified complexity as one of the objectives
    return (lhs.fitness == rhs.fitness &&
            lhs.fitness.complexity == rhs.fitness.complexity);
}

template<ProgramType T>
bool Archive<T>::sameObjectives(const Individual<T>& lhs, 
        const Individual<T>& rhs)
{
    return (lhs.fitness == rhs.fitness);

}

template<ProgramType T>
void Archive<T>::init(Population<T>& pop) 
{
    // TODO: copy the population to a new vector (instead of changing inplace).
    // also, fix this in update function

    individuals.resize(0);

    // dealing with islands --> fast nds for each island
    for (int island =0; island< pop.num_islands; ++island) {
        vector<size_t> indices = pop.get_island_indexes(island);

        selector.fast_nds(pop, indices); 
    }

    // OBS: fast_nds will change all individual fitness inplace.
    // It will update the values for dcounter, rank, and dominated individuals.

    // TODO: fix this way of getting pareto front (the pareto front of different islands combined will not necessarily be the final pareto front). Also fix this in update

    /* vector<size_t> front = this->sorted_front(); */
    for (int island =0; island< pop.num_islands; ++island) {
        auto indices = pop.get_island_indexes(island);

        for (unsigned i = 0; i<indices.size(); ++i)
        {
            const auto& t = *pop.individuals.at(indices.at(i));

            if (t.fitness.rank ==1){
                // we can store a reference for the original ind, since
                // variation operators does not change inplace. Ideally, the
                // original individual is modified inplace just by fit(), which
                // is a side effect that is OK to have here
                individuals.push_back(t);
            }
        } 
    }
    if (this->sort_complexity)
        std::sort(individuals.begin(),individuals.end(), &sortComplexity); 
    else
        std::sort(individuals.begin(),individuals.end(), &sortObj1); 

}

template<ProgramType T>
void Archive<T>::update(Population<T>& pop, const Parameters& params)
{
    individuals.resize(0);  // clear archive

    // refill archive with new pareto fronts (one pareto front for each island!)
    // TODO: refill with fast nds just like hall of fame
    for (int island =0; island< pop.num_islands; ++island) {
        vector<size_t> indices = pop.get_island_indexes(island);

        // TODO: can i just call fast nds with all indexes in indices?
        vector<vector<int>> front = selector.fast_nds(pop, indices); 
        for (const auto& i : front[0])
        {
            individuals.push_back( *pop.individuals.at(i) );
        }
    }
    
    if (this->sort_complexity) {
        std::sort(individuals.begin(), individuals.end(), &sortComplexity); 
    }
    else {
        std::sort(individuals.begin(), individuals.end(), &sortObj1); 
    }
    
    /* auto it = std::unique(individuals.begin(),individuals.end(), &sameFitComplexity); */
    auto it = std::unique(individuals.begin(),individuals.end(), 
            &sameObjectives);

    individuals.resize(std::distance(individuals.begin(),it));
}

} // Pop
} // Brush