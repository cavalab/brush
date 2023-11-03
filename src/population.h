#ifndef POPULATION_H
#define POPULATION_H

#include "program/program.h"
#include "search_space.h"
#include "individual.h"

using std::vector;
using std::string;
using Eigen::Map;

namespace Brush {   
namespace Pop {

template<ProgramType T> 
class Population{
public:        
    vector<Individual<T>*> individuals; 
    Population(int p=0);
    ~Population();

    // fitness, objetives, complexity, etc
    // setters and getters
    // wrappers (fit, predict). This class should also have its own cpp wrapper
};

}// Pop
}// Brush

#endif
