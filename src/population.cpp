/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "population.h"

namespace Brush{   
namespace Pop{
        
template<Brush::ProgramType T>
Population<T>::Population(int p)
{
   individuals.resize(p);
}

template<Brush::ProgramType T>
void Population<T>::init(const SearchSpace& ss, const Parameters& params)
{
   // TODO: load file (like feat)

    #pragma omp parallel for
    for (int i = 0; i< individuals.size(); ++i)
    {          
        individuals.at(i).init(ss, params, i);
    }
}

} // Pop
} // Brush
