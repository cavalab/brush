#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "program/program.h"

namespace Brush{
namespace Pop{
    
template<ProgramType T> 
class Individual{
private:
    Program<T> program; ///< executable data structure

    // store just info that we dont have a getter. size, depth, complexity: they can all be obtained with program.<function here>

    VectorXf error;     ///< training error (used in lexicase selectors)

    float fitness;     ///< aggregate fitness score
    float fitness_v;   ///< aggregate validation fitness score

    unsigned int dcounter;  ///< number of individuals this dominates

    vector<unsigned int> dominated; ///< individual indices this dominates
    unsigned int rank;             ///< pareto front rank
    float crowd_dist;   ///< crowding distance on the Pareto front

public:        
    Individual();

    // fitness, objetives, complexity, etc
    void fit(Dataset& data) { program.fit(data); };
    auto predict(Dataset& data) { return program.predict(data); };

    // TODO: predict proba and classification related methods.

    // setters and getters

    // TODO: This class should also have its own cpp wrapper. Update it into the deap api (the idea is that the user is still able to prototype with brush, I dont think we should disable that feature)

    void init(const SearchSpace& ss, const Parameters& params);

    // getters
    string get_model() { return program.get_model(); };
    size_t get_size() { return program.size(); };
    size_t get_depth() { return program.depth(); };
    size_t get_complexity() { return program.complexity(); };
};

} // Pop
} // Brush

#endif
