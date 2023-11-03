/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "individual.h"

namespace Brush{   
namespace Pop{ 
           
template<ProgramType T> 
Individual<T>::Individual()
{
    // TODO: calculate this stuff
    fitness = -1;
    fitness_v = -1;
    
    dcounter=-1;
    crowd_dist = -1;
}

template<ProgramType T> 
void Individual<T>::initialize(const SearchSpace& ss, const Parameters& params)
{
    // TODO: make searchspace use params, so it will generate something valid
    program = SS.make_program<T>(params.max_depth, params.max_size);
}

} // Pop
} // Brush
