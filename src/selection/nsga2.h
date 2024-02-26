#ifndef NSGA2_H
#define NSGA2_H

#include "selection_operator.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;
using namespace Sel;

template<ProgramType T> 
class NSGA2 : public SelectionOperator<T>
{
public:
    // should operate only on a given island index
    /** NSGA-II based selection and survival methods. */

    // if any of the islands have overlapping indexes, parallel access and modification should be ok (because i dont increase or decrease pop size, not change island ranges inside selection)

    NSGA2(bool surv=false);
    ~NSGA2(){};

    /// selection according to the survival scheme of NSGA-II
    vector<size_t> select(Population<T>& pop, int island,
            const Parameters& p);
    
    /// survival according to the survival scheme of NSGA-II
    vector<size_t> survive(Population<T>& pop, int island, 
            const Parameters& p);
    
    //< Fast non-dominated sorting
    vector<vector<int>> fast_nds(Population<T>&, vector<size_t>&);                

    // front cannot be an attribute because selection will be executed in different threads for different islands (this is a modificationf rom original FEAT code that I got inspiration)

    //< crowding distance of a front i
    void crowding_distance(Population<T>&, vector<vector<int>>&, int); 
        
    private:
        /// sort based on rank, breaking ties with crowding distance
        struct sort_n 
        {
            const Population<T>& pop;          ///< population address

            sort_n(const Population<T>& population) : pop(population) {};

            bool operator() (int i, int j) {
                // TODO: Improve operator[], and decrease use of pop.individuals.at(). Also, decrease number of auto declarations
                auto ind1 = pop.individuals[i];
                auto ind2 = pop.individuals[j];
                
                if (ind1->fitness.rank < ind2->fitness.rank)
                    return true;
                else if (ind1->fitness.rank == ind2->fitness.rank &&
                            ind1->fitness.crowding_dist > ind2->fitness.crowding_dist)
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

            bool operator() (int i, int j) { 
                return pop[i].fitness.get_wvalues()[m] < pop[j].fitness.get_wvalues()[m]; };
        };
    
        size_t tournament(Population<T>& pop, size_t i, size_t j) const;
};

} // selection
} // Brush
#endif