#include "population.h"

namespace Brush{   
namespace Pop{
        
template<ProgramType T>
Population<T>::Population()
{
    individuals.resize(0);
    mig_prob = 0.0;
    pop_size = 0;
    n_islands = 0;
}

template<ProgramType T>
void Population<T>::init(SearchSpace& ss, const Parameters& params)
{
    this->mig_prob = params.mig_prob;
    this->pop_size = params.pop_size;
    this->n_islands=params.num_islands;
    
    // Tuples with start and end indexes for each island. Number of individuals
    // in each island can slightly differ if N_ISLANDS is not a divisor of p (popsize)
    island_indexes.resize(n_islands);
    
    size_t p = pop_size; // population size

    for (int i=0; i<n_islands; ++i)
    {
        size_t idx_start = std::floor(i*p/n_islands);
        size_t idx_end = std::floor((i+1)*p/n_islands);

        auto delta = idx_end - idx_start;

        island_indexes.at(i).resize(delta);
        iota(island_indexes.at(i).begin(), island_indexes.at(i).end(), idx_start);
    };

   // TODO: load file (like feat)

    // this calls the default constructor for the container template class 
    individuals.resize(2*p); // we will never increase or decrease the size during execution (because is not thread safe). this way, theres no need to sync between selecting and varying the population

    #pragma omp parallel for
    for (int i = 0; i< p; ++i)
    {          
        individuals.at(i) = std::make_shared<Individual<T>>();
        individuals.at(i)->init(ss, params);
    }
}

/// update individual vector size and island indexes
template<ProgramType T>
void Population<T>::prep_offspring_slots(int island)
{	   
    // reading and writing is thread-safe, as long as there's no overlap on island ranges.
    // manipulating a vector IS NOT thread-safe (inserting and erasing elements).
    // So, prep_offspring_slots and update should be the synchronization points, not 
    // operations performed concurrently

    size_t p = pop_size; // population size. prep_offspring slots will douple the population, adding the new expressions into the islands
    
    // this is going to be tricky (pay attention to delta and p use)
    size_t idx_start = std::floor(island*p/n_islands);
    size_t idx_end   = std::floor((island+1)*p/n_islands);

    auto delta = idx_end - idx_start;

    // inserting indexes of the offspring
    island_indexes.at(island).resize(delta*2);
    iota(
        island_indexes.at(island).begin() + p, island_indexes.at(island).end(),
        p+idx_start);

    // Im keeping the offspring and parents in the same population object, because we
    // have operations that require them together (archive, hall of fame.)
    // The downside is having to be aware that islands will create offsprings
    // intercalated with other islands
}

template<ProgramType T>
void Population<T>::update(vector<vector<size_t>> survivors)
{
    vector<std::shared_ptr<Individual<T>>> new_pop;
    new_pop.resize(pop_size);
    size_t i=0;
    for (int j=0; j<n_islands; ++j)
    {
        for (int k=0; k<survivors.at(j).size(); ++k){
            new_pop.at(i) = individuals.at(survivors.at(j).at(k));
            
            // update will set the complexities (for migration step. we do it here because update handles non-thread safe operations)
            new_pop.at(i)->set_complexity();
    
            ++i;
        }

        // need to make island point to original range
        size_t idx_start = std::floor(j*size/n_islands);
        size_t idx_end   = std::floor((j+1)*size/n_islands);

        auto delta = idx_end - idx_start;

        // inserting indexes of the offspring
        island_indexes.at(j).resize(delta);
        iota(island_indexes.at(j).begin(), island_indexes.at(j).end(), idx_start);
    }
    individuals = new_pop;
}

template<ProgramType T>
string Population<T>::print_models(bool just_offspring, string sep)
{
    // not printing the island each individual belongs to
    string output = "";

    int start = 0;
   
   if (just_offspring)
       start = individuals.size()/2;

   for (unsigned int i=start; i< individuals.size(); ++i)
       output += individuals.at(i)->get_model() + sep;
   
   return output;
}

template<ProgramType T>
vector<vector<size_t>> Population<T>::sorted_front(unsigned rank)
{
    // this is used to migration and update archive at the end of a generation. expect islands without offspring

    /* Returns individuals on the Pareto front, sorted by increasign complexity. */
    vector<vector<size_t>> pf_islands;
    pf_islands.resize(n_islands);

    for (int i=0; i<n_islands; ++i)
    {
        auto idxs = island_indexes.at(i);
        vector<size_t> pf;

        for (unsigned int& i : idxs)
        {
            // this assumes that rank was previously calculated. It is set in selection (ie nsga2) if the information is useful to select/survive
            if (individuals.at(i)->rank == rank)
                pf.push_back(i);
        }
        std::sort(pf.begin(),pf.end(),SortComplexity(*this)); 
        auto it = std::unique(pf.begin(),pf.end(),SameFitComplexity(*this));
        
        pf.resize(std::distance(pf.begin(),it));
        pf_islands.at(i) = pf;
    }

    return pf_islands;
}

template<ProgramType T>
vector<size_t> Population<T>::hall_of_fame(unsigned rank)
{
    // this is used to migration and update archive at the end of a generation. expect islands without offspring

    vector<size_t> pf(0);
    for (unsigned int i =0; i<individuals.size(); ++i)
    {
        if (individuals.at(i)->rank == rank)
            pf.push_back(i);
    }
    std::sort(pf.begin(),pf.end(),SortComplexity(*this)); 
    auto it = std::unique(pf.begin(),pf.end(),SameFitComplexity(*this));
    pf.resize(std::distance(pf.begin(),it));

    return pf;
}


template<ProgramType T>
void Population<T>::migrate()
{
    // changes where island points to 

    if (n_islands==1)
        return;

    auto island_fronts = sorted_front();
    auto global_hall_of_fame = hall_of_fame();

    // This is not thread safe (as it is now)
    for (int island=0; island<n_islands; ++island)
    {
        auto idxs = island_indexes.at(island);
        for (unsigned int i=0; i<idxs.size(); ++i)
        {
            if (r() < mig_prob)
            {
                size_t migrating_idx;
                // determine if incoming individual comes from global or local hall of fame
                if (r() < 0.5) { // from global hall of fame
                    migrating_idx = *r.select_randomly(
                        global_hall_of_fame.begin(),
                        global_hall_of_fame.end());
                }
                else { // from any other local hall of fame
                    // finding other island indexes
                    vector<int> other_islands(n_islands-1);
                    iota(other_islands.begin(), other_islands.end(), 0);

                    // skipping current island
                    auto it = other_islands.begin();
                    std::advance(it, island);
                    for (;it != other_islands.end(); ++it) {
                        ++(*it);
                    }

                    // picking other island
                    int other_island = *r.select_randomly(
                        other_islands.begin(),
                        other_islands.end());

                    migrating_idx = *r.select_randomly(
                        island_fronts.at(other_island).begin(),
                        island_fronts.at(other_island).end());
                }
                
                island_indexes.at(i) = migrating_idx;
            }
        }
    }
}

} // Pop
} // Brush
