#include "nsga2.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;
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

    int flag = ind1.fitness.dominates(ind2.fitness);
    
    if (flag == 1) // ind1 dominates ind2
        return i;
    else if (flag == -1) // ind2 dominates ind1
        return j;
    else if (ind1.crowding_dist > ind2.crowding_dist)
        return i;
    else if (ind2.crowding_dist > ind1.crowding_dist)
        return j;
    else 
        return i; 
}

template<ProgramType T>
vector<size_t> NSGA2<T>::select(Population<T>& pop, int island, 
        const Parameters& params)
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
        const Parameters& params)
{

    // fmt::print("starting\n");
    size_t idx_start = std::floor(island*pop.size()/pop.num_islands);
    size_t idx_end   = std::floor((island+1)*pop.size()/pop.num_islands);

    int original_size = (idx_end - idx_start)/2; // original island size (survive must   be  called with an island with offfspring)

    auto island_pool = pop.get_island_indexes(island);

    // fmt::print("indexes {} {}\n", idx_start, idx_end);
    
    // set objectives (this is when the obj vector is updated.)
    
    // for loop below (originally performed in selection in FEAT) was moved to evaluation --- multiple islands may have the same individual
    // for (unsigned int i=0; i<island_pool.size(); ++i)
    //     pop.individuals.at(island_pool[i])->set_obj(params.objectives);

    // fast non-dominated sort
    // fmt::print("fast nds\n");
    auto front = fast_nds(pop, island_pool);
    
    // fmt::print("selecting...\n");
    // Push back selected individuals until full
    vector<size_t> selected;
    // fmt::print("created array...\n");
    selected.resize(0);
    // fmt::print("resized...\n");
    
    int i = 0;

    // fmt::print("starting loop...\n");
    // fmt::print("{}...\n",selected.size());
    // fmt::print("{}...\n", front.at(i).size());
    // fmt::print("{}...\n", original_size);

    while (
        i < front.size()
        && ( selected.size() + front.at(i).size() < original_size ) // (size/2) because we want to get to the original size (prepare_offspring_slots doubled it before survival operation)
    )
    {
        // fmt::print("1...\n");
        std::vector<int>& Fi = front.at(i);        // indices in front i

        // fmt::print("2...\n");
        crowding_distance(pop, front, i);          // calculate crowding in Fi

        // fmt::print("3...\n");    
        for (int j = 0; j < Fi.size(); ++j)     // Pt+1 = Pt+1 U Fi
            selected.push_back(Fi.at(j));
        
        // fmt::print("4...\n");  

        ++i;
    }

    // fmt::print("crowding distance\n");
    crowding_distance(pop, front, i);   // calculate crowding in final front to include
    std::sort(front.at(i).begin(),front.at(i).end(),sort_n(pop));
    
    // fmt::print("adding last front)\n");
    const int extra = original_size - selected.size();
    for (int j = 0; j < extra; ++j) // Pt+1 = Pt+1 U Fi[1:N-|Pt+1|]
        selected.push_back(front.at(i).at(j));
    
    // fmt::print("returning\n");
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
        
            int compare = p->fitness.dominates(q.fitness);
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
            p->fitness.dcounter  = dcount;
            p->fitness.dominated.clear();
            p->fitness.dominated = dom; // dom will have values already referring to island indexes
        
            if (p->dcounter == 0) {
                p->fitness.set_rank(1);
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
            for (int j = 0; j < p.fitness.dominated.size() ; ++j) {

                auto q = pop.individuals.at(p.fitness.dominated.at(j));
                q->fitness.dcounter -= 1;

                if (q->fitness.dcounter == 0) {
                    q->fitness.set_rank(fi+1);
                    Q.push_back(p.fitness.dominated.at(j));
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
        pop.individuals.at(F.at(i))->fitness.crowding_dist = 0;

    const int limit = pop.individuals.at(0)->fitness.get_wvalues().size();
    for (int m = 0; m < limit; ++m) {

        std::sort(F.begin(), F.end(), comparator_obj(pop,m));

        // in the paper dist=INF for the first and last, in the code
        // this is only done to the first one or to the two first when size=2
        pop.individuals.at(F.at(0))->fitness.crowding_dist = std::numeric_limits<float>::max();
        if (fsize > 1)
            pop.individuals.at(F.at(fsize-1))->fitness.crowding_dist = std::numeric_limits<float>::max();
    
        for (int i = 1; i < fsize-1; ++i) 
        {
            if (pop.individuals.at(F.at(i))->fitness.crowding_dist != std::numeric_limits<float>::max()) 
            {   // crowd over obj
                pop.individuals.at(F.at(i))->fitness.crowding_dist +=
                    (pop.individuals.at(F.at(i+1))->fitness.get_wvalues().at(m) - pop.individuals.at(F.at(i-1))->fitness.get_wvalues().at(m)) 
                    / (pop.individuals.at(F.at(fsize-1))->fitness.get_wvalues().at(m) - pop.individuals.at(F.at(0))->fitness.get_wvalues().at(m));
            }
        }
    }        
}

} // selection
} // Brush