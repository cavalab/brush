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
    else if (ind1.fitness.crowding_dist > ind2.fitness.crowding_dist)
        return i;
    else if (ind2.fitness.crowding_dist > ind1.fitness.crowding_dist)
        return j;
    else 
        return i; 
}

template<ProgramType T>
vector<size_t> NSGA2<T>::select(Population<T>& pop, int island, 
        const Parameters& params)
{
    // tournament selection.
    
    // TODO: move this to tournament selection file, and throw not implemented error in nsga.
    auto island_pool = pop.get_island_indexes(island);

    // if this is first generation, just return indices to pop
    if (params.current_gen==0)
        return island_pool;

    // i am not sure if I need this update of rank and crowding distance (bc first generation is ignored by if above, and the other generations will always have individuals that went through survival, which already calculates this information. TODO: in the final algorithm, I need to make sure this is correct)
    auto front = fast_nds(pop, island_pool);
    for (size_t i = 0; i< front.size(); i++)
    {
        crowding_distance(pop, front, i);
    }

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
    // TODO: clean up comments mess here 

    // combining all islands to get the pareto fronts
    std::vector<size_t> island_pool;
    for (int i = 0; i < params.num_islands; ++i) {
        auto indexes = pop.get_island_indexes(i);
        island_pool.insert(island_pool.end(), indexes.begin(), indexes.end());
    }
    
    // fast non-dominated sort
    auto front = fast_nds(pop, island_pool);
    
    // Push back selected individuals until full
    vector<size_t> selected;
    selected.resize(0);
    
    int i = 0;
    while (
        i < front.size()
        && ( selected.size() + front.at(i).size() < params.pop_size )
    )
    {
        std::vector<int>& Fi = front.at(i);        // indices in front i

        crowding_distance(pop, front, i);          // calculate crowding in Fi

        for (int j = 0; j < Fi.size(); ++j)     // Pt+1 = Pt+1 U Fi
            selected.push_back(Fi.at(j));

        ++i;
    }

    // fmt::print("crowding distance\n");
    crowding_distance(pop, front, i);   // calculate crowding in final front to include
    std::sort(front.at(i).begin(),front.at(i).end(),sort_n(pop));
    
    // fmt::print("adding last front)\n");
    const int extra = params.pop_size - selected.size();
    for (int j = 0; j < extra; ++j) // Pt+1 = Pt+1 U Fi[1:N-|Pt+1|]
        selected.push_back(front.at(i).at(j));
    
    // fmt::print("returning\n");
    return selected;
}

template<ProgramType T>
vector<vector<int>> NSGA2<T>::fast_nds(Population<T>& pop, vector<size_t>& island_pool) 
{
    // this will update pareto dominance attributes in fitness class
    // based on the population

    //< the Pareto fronts
    vector<vector<int>> front;                

    front.resize(1);
    front.at(0).clear();

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
        p->fitness.dcounter = dcount;
        p->fitness.dominated = dom; // dom will have values already referring to island indexes
        // p->fitness.set_crowding_dist(0.0f);
    
        if (p->fitness.dcounter == 0) {
            // fmt::print("pushing {}...\n", island_pool[i]);
            p->fitness.set_rank(1);
            // front will have values already referring to island indexes
            front.at(0).push_back(island_pool[i]);
        }

    }

    // fmt::print("First front size {}...\n", front.at(0).size());
    
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
            for (int j = 0; j < p.fitness.dominated.size(); ++j) {
                // fmt::print("decreased counter of ind {} for {} to {} \n", j, p.fitness.dominated.at(j), pop.individuals.at(p.fitness.dominated.at(j))->fitness.dcounter);

                auto q = pop.individuals.at(p.fitness.dominated.at(j));
                
                // fmt::print("decreased counter \n");
                q->fitness.dcounter -= 1;

                if (q->fitness.dcounter == 0) {
                    // fmt::print("updated counter for ind {} \n", j);

                    q->fitness.set_rank(fi+1);
                    Q.push_back(p.fitness.dominated.at(j));
                }
            }
        }

        front.push_back(Q);

        fi += 1;
    }
    return front;
}

template<ProgramType T>
void NSGA2<T>::crowding_distance(Population<T>& pop, vector<vector<int>>& front, int fronti)
{

    // fmt::print("inside crowding distance for front {}...\n", fronti);

    std::vector<int> F = front.at(fronti);
    if (F.size() == 0 ){
        // fmt::print("empty front\n");
        return;
    }

    const int fsize = F.size();
    // fmt::print("front size is {}...\n", fsize);

    for (int i = 0; i < fsize; ++i)
        pop.individuals.at(F.at(i))->fitness.set_crowding_dist(0.0f);

    // fmt::print("reseted crowding distance for individuals in this front\n");

    const int limit = pop.individuals.at(0)->fitness.get_wvalues().size();
    // fmt::print("limit is {}\n", limit);

    for (int m = 0; m < limit; ++m) {
        // fmt::print("m {}\n", m);

        std::sort(F.begin(), F.end(), comparator_obj(pop,m));

        // in the paper dist=INF for the first and last, in the code
        // this is only done to the first one or to the two first when size=2
        pop.individuals.at(F.at(0))->fitness.crowding_dist = std::numeric_limits<float>::max();
        if (fsize > 1)
            pop.individuals.at(F.at(fsize-1))->fitness.crowding_dist = std::numeric_limits<float>::max();
    
        float first_of_front = pop.individuals.at(F.at(0))->fitness.get_wvalues().at(m);
        float last_of_front  = pop.individuals.at(F.at(fsize-1))->fitness.get_wvalues().at(m);
        for (int i = 1; i < fsize-1; ++i) 
        {
            if (pop.individuals.at(F.at(i))->fitness.crowding_dist != std::numeric_limits<float>::max())
            {
                float next_of_front = pop.individuals.at(F.at(i+1))->fitness.get_wvalues().at(m);
                float prev_of_front = pop.individuals.at(F.at(i-1))->fitness.get_wvalues().at(m);

                // updating the value by aggregating crowd dist for each objective
                pop.individuals.at(F.at(i))->fitness.crowding_dist +=
                    (next_of_front - prev_of_front) / (last_of_front - first_of_front);
            }
        }
    }        
}

} // selection
} // Brush