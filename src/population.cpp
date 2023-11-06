/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "population.h"

namespace Brush{   
namespace Pop{
        

template<Brush::ProgramType T>
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

template<Brush::ProgramType T>
Population<T>::Population(int p, int n_islands)
{
    individuals.resize(p);

    this->n_islands=n_islands;
    set_island_ranges();

    island_skip.resize(n_islands);
    iota(island_skip.begin(), island_skip.end(), 0);

    offspring_ready = false;
}

template<Brush::ProgramType T>
void Population<T>::init(const SearchSpace& ss, const Parameters& params)
{
   // TODO: load file (like feat)
    #pragma omp parallel for
    for (int i = 0; i< individuals.size(); ++i)
    {          
        individuals.at(i).init(ss, params);
    }
}

/// update individual vector size and island indexes
template<Brush::ProgramType T>
void Population<T>::prep_offspring_slots()
{	   
    if (offspring_ready)
        HANDLE_ERROR_THROW("Allocating space in population that already has active offspring slots");

    vector<Individual<T>*> expanded_pop;
    expanded_pop.resize(2*individuals.size());

    for (int i=0; i<n_islands; ++i)
    {
        // old indexes
        auto [idx_start, idx_end] = island_ranges.at(i);
        size_t delta = idx_end - idx_start;
        
        for (int j=0; j<delta; j++) {
            expanded_pop.at(2*idx_start + j) = individuals.at(idx_start+j);
        }
        
        // setting new island sizes (TODO: i think I can just call set island
        // ranges again, but i need to do the math to see if floor operations
        // will not accidentally migrate some individuals)
        island_ranges.at(i) = {2*idx_start, 2*(idx_end + delta)};
    };

    this->individuals = &expanded_pop;
    offspring_ready = true;
}

template<Brush::ProgramType T>
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

template<Brush::ProgramType T>
string Population<T>::print_models(bool just_offspring, string sep)
{
    // not printing the island each individual belongs to
    string output = "";

    for (int i=0; i<n_islands; ++i)
    {
        auto [idx_start, idx_end] = island_ranges.at(i);
        size_t skip = island_skip.at(i); // number of individuals to ignore because variation failed
        
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

template<Brush::ProgramType T>
vector<vector<size_t>> Population<T>::sorted_front(unsigned rank)
{
    // this is used to update archive at the end of a generation. Supose islands without offspring

    /* Returns individuals on the Pareto front, sorted by increasign complexity. */
    vector<vector<size_t>> pf_islands;
    pf_islands.resize(n_islands);

    for (int i=0; i<n_islands; ++i)
    {
        auto [idx_start, idx_end] = island_ranges.at(i);
        vector<size_t> pf;

        for (unsigned int i =idx_start; i<idx_end; ++i)
        {
            // this assumes that rank was previously calculated
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


} // Pop
} // Brush
