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

    size_t complexity;
    unsigned int dcounter;  ///< number of individuals this dominates
    vector<unsigned int> dominated; ///< individual indices this dominates
    
    unsigned int rank;             ///< pareto front rank
    float crowd_dist;   ///< crowding distance on the Pareto front

public:        
    Individual()
    { // TODO: calculate this stuff
        fitness = -1;
        fitness_v = -1;
        
        complexity=-1;
        dcounter=-1;
        rank=-1;
        crowd_dist = -1;
    };

    void init(const SearchSpace& ss, const Parameters& params)
    {
        program = SS.make_program<T>(params, 0, 0);

        // If different from zero, then the program is created with a fixed depth and size.
        // If zero, it samples the value
        // program = SS.make_program<T>(params, params.max_depth, params.max_size);
    };

    // fitness, objetives, complexity, etc
    void fit(Dataset& data) { program.fit(data); };
    auto predict(Dataset& data) { return program.predict(data); };

    // TODO: predict proba and classification related methods.
    // TODO: This class should also have its own cpp wrapper. Update it into the deap api (the idea is that the user is still able to prototype with brush, I dont think we should disable that feature)

    // just getters
    string get_model() { return program.get_model(); };
    size_t get_size() { return program.size(); };
    size_t get_depth() { return program.depth(); };

    // setters and getters
    size_t set_complexity() {
        complexity = program.complexity();
        return complexity;
    }; // sets and returns it
    size_t get_complexity() const { return complexity; };

    void set_rank(unsigned r){ rank=r; };
    size_t get_rank() const { return rank; };

    void set_crowd_dist(unsigned cd){ crowd_dist=cd; };
    size_t get_crow_dist() const { return crowd_dist; };
};

} // Pop
} // Brush

#endif
