/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "population.h"

namespace Brush{   
namespace Pop{
        
int last; 

template<Brush::ProgramType T>
Population<T>::Population(int p)
{
   individuals.resize(p);
}

template<Brush::ProgramType T>
Population<T>::~Population(){}

} // Pop
} // Brush
