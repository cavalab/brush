#ifndef NSGA2_H
#define NSGA2_H

#include "selection.h"
#include "../init.h"
#include "../program/program.h"
#include "../population.h"
#include "../individual.h"
#include "../data/data.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;
using namespace Data;
using namespace Sel;

template<ProgramType T> 
class NSGA2 : public SelectionOperator
{
    /** NSGA-II based selection and survival methods. */

    NSGA2(bool surv);
    ~NSGA2();

    /// selection according to the survival scheme of NSGA-II
    vector<size_t> select(Population<T>& pop,  
            const Parameters& p, const Dataset& d);
    
    /// survival according to the survival scheme of NSGA-II
    vector<size_t> survive(Population<T>& pop,  
            const Parameters& p, const Dataset& d);
    
    //< the Pareto fronts
    vector<vector<int>> front;                

    //< Fast non-dominated sorting
    void fast_nds(vector<Individual<T>>&);                

    //< crowding distance of a front i
    void crowding_distance(Population<T>&, int); 
        
    private:
        /// sort based on rank, breaking ties with crowding distance
        struct sort_n 
        {
            const Population<T>& pop;          ///< population address
            sort_n(const Population<T>& population) : pop(population) {};
            bool operator() (int i, int j) {
                const Individual<T>& ind1 = pop.individuals[i];
                const Individual<T>& ind2 = pop.individuals[j];
                if (ind1.rank < ind2.rank)
                    return true;
                else if (ind1.rank == ind2.rank &&
                            ind1.crowd_dist > ind2.crowd_dist)
                    return true;
                return false;
            };
        };

        /// sort based on objective m
        struct comparator_obj 
        {
            const Population<T>& pop;      ///< population address
            int m;                      ///< objective index 
            comparator_obj(const Population<T>& population, int index) 
                : pop(population), m(index) {};
            bool operator() (int i, int j) { return pop[i].obj[m] < pop[j].obj[m]; };
        };
    
        size_t tournament(vector<Individual<T>>& pop, size_t i, size_t j) const;
};

} // selection
} // Brush
#endif