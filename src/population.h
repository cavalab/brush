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
    ~Population(){};

    void init(const SearchSpace& ss, const Parameters& params);
};

}// Pop
}// Brush

#endif
