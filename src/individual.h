#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "program/program.h"

namespace Brush{
namespace Pop{
    
template<ProgramType T> 
class Individual{
public:        
    Program<T> program; ///< executable data structure

    Individual(Program<T> Prog);

    // fitness, objetives, complexity, etc
    // setters and getters
    // wrappers (fit, predict). This class should also have its own cpp wrapper
};

} // Pop
} // Brush

#endif
