/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "individual.h"

namespace Brush{   
namespace Pop{ 
           
template<ProgramType T> 
Individual<T>::Individual(Program<T> Prog)
{
    program = Prog;

    // TODO: calculate this stuff
    complexity = -1; 
    fitness = -1;
    fitness_v = -1;
    fairness = -1;
    fairness_v = -1;
    dcounter=-1;
    crowd_dist = -1;
}


// void Individual::initialize(const Parameters& params, bool random, int id)
// {

// }

} // Pop
} // FT
