#include "nsga2.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;
using namespace Data;
using namespace Sel;

template<ProgramType T>
NSGA2<T>::NSGA2(bool surv)
{ 
    this->name = "nsga2"; 
    this->survival = surv; 
}

template<ProgramType T>
size_t NSGA2<T>::tournament(Population<T>& pop, size_t i, size_t j) const 
{
    // gets two individuals and compares them. i and j bhould be within island range
    const Individual<T>& ind1 = pop[i];
    const Individual<T>& ind2 = pop[j];

    int flag = ind1.check_dominance(ind2);
    
    if (flag == 1) // ind1 dominates ind2
        return i;
    else if (flag == -1) // ind2 dominates ind1
        return j;
    else if (ind1.crowd_dist > ind2.crowd_dist)
        return i;
    else if (ind2.crowd_dist > ind1.crowd_dist)
        return j;
    else 
        return i; 
}

template<ProgramType T>
vector<size_t> NSGA2<T>::select(Population<T>& pop, int island, 
        const Parameters& params, const Dataset& d)
{
    auto island_pool = pop.get_island_indexes(island);

    // if this is first generation, just return indices to pop
    if (params.current_gen==0)
        return island_pool;

    // setting the objectives
    // for (unsigned int i=0; i<island_pool.size(); ++i)
    //     pop.individuals.at(island_pool[i])->set_obj(params.objectives);

    vector<size_t> selected(0); 
    for (int i = 0; i < island_pool.size(); ++i) // selecting based on island_pool size
    {
        size_t winner = tournament(pop,
            *r.select_randomly(island_pool.begin(), island_pool.end()), 
            *r.select_randomly(island_pool.begin(), island_pool.end()));
        
        selected.push_back(winner);
    }
    return selected;
}

template<ProgramType T>
vector<size_t> NSGA2<T>::survive(Population<T>& pop, int island,
        const Parameters& params, const Dataset& d)
{

    size_t idx_start = std::floor(island*pop.size()/pop.n_islands);
    size_t idx_end   = std::floor((island+1)*pop.size()/pop.n_islands);

    auto original_size = idx_end - idx_start; // island size

    auto island_pool = pop.get_island_indexes(island);

    // set objectives (this is when the obj vector is updated.)
    
    // for loop below (originally performed in selection in FEAT) was moved to evaluation --- multiple islands may have the same individual
    // for (unsigned int i=0; i<island_pool.size(); ++i)
    //     pop.individuals.at(island_pool[i])->set_obj(params.objectives);

    // fast non-dominated sort
    auto front = fast_nds(pop, island_pool);
    
    // Push back selected individuals until full
    vector<size_t> selected(0);
    int i = 0;
    while ( selected.size() + front.at(i).size() < original_size ) // (size/2) because we want to get to the original size (prepare_offspring_slots doubled it before survival operation)
    {
        std::vector<int>& Fi = front.at(i);        // indices in front i
        crowding_distance(pop, front, i);          // calculate crowding in Fi
        
        for (int j = 0; j < Fi.size(); ++j)     // Pt+1 = Pt+1 U Fi
            selected.push_back(Fi.at(j));
        
        ++i;
    }

    crowding_distance(pop, front, i);   // calculate crowding in final front to include
    std::sort(front.at(i).begin(),front.at(i).end(),sort_n(pop));
    
    fmt::print("adding last front)\n");
    const int extra = original_size - selected.size();
    for (int j = 0; j < extra; ++j) // Pt+1 = Pt+1 U Fi[1:N-|Pt+1|]
        selected.push_back(front.at(i).at(j));
    
    return selected;
}

template<ProgramType T>
vector<vector<int>> NSGA2<T>::fast_nds(Population<T>& pop, vector<size_t>& island_pool) 
{
    //< the Pareto fronts
    vector<vector<int>> front;                

    front.resize(1);
    front.at(0).clear();

    #pragma omp parallel for
    for (int i = 0; i < island_pool.size(); ++i) {
    
        std::vector<unsigned int> dom;
        int dcount = 0;
    
        auto p = pop.individuals.at(island_pool[i]);

        for (int j = 0; j < island_pool.size(); ++j) {
        
            const Individual<T>& q = pop[island_pool[j]];
        
            int compare = p->check_dominance(q);
            if (compare == 1) { // p dominates q
                //p.dominated.push_back(j);
                dom.push_back(island_pool[j]);
            } else if (compare == -1) { // q dominates p
                //p.dcounter += 1;
                dcount += 1;
            }
        }
    
        #pragma omp critical
        {
            p->dcounter  = dcount;
            p->dominated.clear();
            p->dominated = dom; // dom will have values already referring to island indexes
        
            if (p->dcounter == 0) {
                p->set_rank(1);
                // front will have values already referring to island indexes
                front.at(0).push_back(island_pool[i]);
            }
        }
    }
    
    // using OpenMP can have different orders in the front.at(0)
    // so let's sort it so that the algorithm is deterministic
    // given a seed
    std::sort(front.at(0).begin(), front.at(0).end());    

    int fi = 1;
    while (front.at(fi-1).size() > 0) {

        std::vector<int>& fronti = front.at(fi-1);
        std::vector<int> Q;
        for (int i = 0; i < fronti.size(); ++i) {

            const Individual<T>& p = pop[fronti.at(i)];

            // iterating over dominated individuals
            for (int j = 0; j < p.dominated.size() ; ++j) {

                auto q = pop.individuals.at(p.dominated.at(j));
                q->dcounter -= 1;

                if (q->dcounter == 0) {
                    q->set_rank(fi+1);
                    Q.push_back(p.dominated.at(j));
                }
            }
        }

        fi += 1;
        front.push_back(Q);
    }

    return front;
}

template<ProgramType T>
void NSGA2<T>::crowding_distance(Population<T>& pop, vector<vector<int>>& front, int fronti)
{
    std::vector<int> F = front.at(fronti);
    if (F.size() == 0 ) return;

    const int fsize = F.size();

    for (int i = 0; i < fsize; ++i)
        pop.individuals.at(F.at(i))->crowd_dist = 0;

    const int limit = pop.individuals.at(0)->obj.size();
    for (int m = 0; m < limit; ++m) {

        std::sort(F.begin(), F.end(), comparator_obj(pop,m));

        // in the paper dist=INF for the first and last, in the code
        // this is only done to the first one or to the two first when size=2
        pop.individuals.at(F.at(0))->crowd_dist = std::numeric_limits<float>::max();
        if (fsize > 1)
            pop.individuals.at(F.at(fsize-1))->crowd_dist = std::numeric_limits<float>::max();
    
        for (int i = 1; i < fsize-1; ++i) 
        {
            if (pop.individuals.at(F.at(i))->crowd_dist != std::numeric_limits<float>::max()) 
            {   // crowd over obj
                pop.individuals.at(F.at(i))->crowd_dist +=
                    (pop.individuals.at(F.at(i+1))->obj.at(m) - pop.individuals.at(F.at(i-1))->obj.at(m)) 
                    / (pop.individuals.at(F.at(fsize-1))->obj.at(m) - pop.individuals.at(F.at(0))->obj.at(m));
            }
        }
    }        
}

} // selection
} // Brush