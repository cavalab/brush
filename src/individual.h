#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "program/program.h"
#include "search_space.h"

namespace Brush{
namespace Pop{
    
template<ProgramType T> 
class Individual{
public: // TODO: make these private (and work with nlohman json)
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
    vector<float> obj; ///< objectives for use with Pareto selection
       
    Individual()
    {
        fitness = -1;
        fitness_v = -1;
        
        complexity=-1;

        dcounter=-1;
        rank=-1;
        crowd_dist = -1;
    };

    Individual(Program<T>& prg) : Individual() { program = prg; };

    void init(SearchSpace& ss, const Parameters& params)
    {
        program = ss.make_program<Program<T>>(params, 0, 0);

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
    Program<T>& get_program() { return program; };

    // setters and getters
    size_t set_complexity() {
        complexity = program.complexity();
        return complexity;
    }; // sets and returns it
    size_t get_complexity() const { return complexity; };

    // TODO: USE setters and getters intead of accessing it directly
    void set_fitness(float f){ fitness=f; };
    float get_fitness() const { return fitness; };

    void set_fitness_v(float f_v){ fitness_v=f_v; };
    float get_fitness_v() const { return fitness_v; };

    void set_rank(unsigned r){ rank=r; };
    size_t get_rank() const { return rank; };

    void set_crowd_dist(unsigned cd){ crowd_dist=cd; };
    size_t get_crowd_dist() const { return crowd_dist; };

    /// set obj vector given a string of objective names
    void set_obj(const vector<string>&); 
    int check_dominance(const Individual<T>& b) const;
};


// serialization for Individual
template<ProgramType T>
void to_json(json &j, const Individual<T> &p)
{
    j = json{
        {"program", p.program},
        {"fitness", p.fitness},
        {"fitness_v", p.fitness_v},
        {"complexity", p.complexity},
        {"rank", p.rank},
        {"crowd_dist", p.crowd_dist}
    }; 
}

template<ProgramType T>
void from_json(const json &j, Individual<T>& p)
{
    j.at("program").get_to( p.program );
    j.at("fitness").get_to( p.fitness );
    j.at("fitness_v").get_to( p.fitness_v );
    j.at("complexity").get_to( p.complexity );
    j.at("rank").get_to( p.rank );
    j.at("crowd_dist").get_to( p.crowd_dist );
}


} // Pop
} // Brush

#endif