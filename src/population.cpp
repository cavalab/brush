/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "population.h"

namespace Brush{   
namespace Pop{
        

template<ProgramType T>
void Population<T>::set_island_ranges()
{
    // everytime we change popsize, this function must be called

    // Tuples with start and end indexes for each island. Number of individuals
    // in each island can slightly differ if N_ISLANDS is not a divisor of p (popsize)
    island_ranges.resize(n_islands);
    
    size_t p = size(); // population size

    for (int i=0; i<n_islands; ++i)
    {
        island_ranges.at(i) = {
            (size_t)std::floor(i*p/n_islands),
            (size_t)std::floor((i+1)*p/n_islands)
        };
    };
}

template<ProgramType T>
Population<T>::Population(int p, int n_islands)
{
    // this calls the default constructor for the container template class 
    individuals.resize(p);

    this->n_islands=n_islands;
    set_island_ranges();

    island_skip.resize(n_islands);
    iota(island_skip.begin(), island_skip.end(), 0);

    offspring_ready = false;
}

template<ProgramType T>
void Population<T>::init(SearchSpace& ss, const Parameters& params)
{
    this->mig_prob = params.mig_prob;

   // TODO: load file (like feat)
    #pragma omp parallel for
    for (int i = 0; i< individuals.size(); ++i)
    {          
        individuals.at(i).init(ss, params);
    }
}

/// update individual vector size and island indexes
template<ProgramType T>
void Population<T>::prep_offspring_slots()
{	   
    // reading and writing is thread-safe, as long as there's no overlap on island ranges.
    // manipulating a vector IS NOT thread-safe (inserting and erasing elements).
    // So, prep_offspring_slots and update should be the synchronization points, not 
    // operations performed concurrently

    // TODO: add _SingleThreaded in funcname
    if (offspring_ready)
        HANDLE_ERROR_THROW("Allocating space in population that already has active offspring slots");

    vector<Individual<T>> expanded_pop;
    expanded_pop.resize(2*individuals.size());

    for (int i=0; i<n_islands; ++i)
    {
        // old indexes
        auto [idx_start, idx_end] = island_ranges.at(i);
        size_t delta = idx_end - idx_start;
        
        for (int j=0; j<delta; j++) {
            expanded_pop.at(2*idx_start + j) = individuals.at(idx_start+j);
        }
        
        // // setting new island sizes (TODO: i think I can just call set island
        // // ranges again, but i need to do the math to see if floor operations
        // // will not accidentally migrate some individuals)
        // island_ranges.at(i) = {2*idx_start, 2*(idx_end + delta)};
    };

    this->individuals = expanded_pop;
    set_island_ranges();
    offspring_ready = true;

    // Im keeping the offspring and parents in the same population object, because we
    // have operations that require them together (archive, hall of fame.)
    // The downside is having to be aware that islands will create offsprings
    // intercalated with other islands
}

template<ProgramType T>
void Population<T>::update(vector<size_t> survivors)
{
    if (!offspring_ready)
        HANDLE_ERROR_THROW("Shrinking a population that has no active offspring");

    assert(survivors.size() == individuals.size()/2 
        && "Cant shrink a population to a size different from the original initial size");

    vector<size_t> pop_idx(individuals.size());
    std::iota(pop_idx.begin(),pop_idx.end(),0);
    std::reverse(pop_idx.begin(),pop_idx.end());
    for (const auto& i : pop_idx)
        if (!in(survivors,i))
            individuals.erase(individuals.begin()+i);                         
    
    set_island_ranges();
    offspring_ready = false;
}

template<ProgramType T>
string Population<T>::print_models(bool just_offspring, string sep)
{
    // not printing the island each individual belongs to
    string output = "";

    for (int i=0; i<n_islands; ++i)
    {
        auto [idx_start, idx_end] = island_ranges.at(i);
        size_t skip = island_skip.at(i); // number of individuals to ignore because variation failed
        //TODO: use taskflow and pragma once correctly (search and fix code)
        if (just_offspring) {
            size_t delta = idx_end - idx_start; // starting from the middle of the island (where the offspring lives)
            idx_start = idx_start + delta/2;
        }
            
        // (linear complexity on size of individuals, even with two nested loops)
        for (int j=idx_start; j<idx_end-skip; j++) {
            output += individuals.at(j).get_model() + sep;
        }
    };

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
        auto [idx_start, idx_end] = island_ranges.at(i);
        vector<size_t> pf;

        for (unsigned int i =idx_start; i<idx_end; ++i)
        {
            // this assumes that rank was previously calculated. It is set in selection (ie nsga2) if the information is useful to select/survive
            if (individuals.at(i).rank == rank)
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
        if (individuals.at(i).rank == rank)
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
    assert(!offspring_ready
        && "pop with offspring dont migrate (run update before calling this)");

    auto island_fronts = sorted_front();
    auto global_hall_of_fame = hall_of_fame();

    // This is not thread safe (as it is now)
    for (int island=0; island<n_islands; ++island)
    {
        auto [idx_start, idx_end] = island_ranges.at(island);
        for (unsigned int i =idx_start; i<idx_end; ++i)
        {
            if (r() < mig_prob)
            {
                size_t migrating_idx;
                // determine if incoming individual comes from global or local hall of fame
                if (r() < 0.5 && n_islands>1) { // from global hall of fame
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
                
                individuals.at(i) = individuals.at(migrating_idx);
            }
        }
    }
}


} // Pop
} // Brush
