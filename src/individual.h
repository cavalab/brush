#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

// #include "search_space.h"
#include "program/program.h"

#include <functional>

using namespace nlohmann;

template <> // this is intended to be used with DEAP. TODO: decide if im going to keep it
struct std::hash<std::vector<float>> {
    std::size_t operator()(const std::vector<float>& v) const {
        std::size_t seed = v.size();
        for (const auto& elem : v) {
            seed ^= std::hash<float>{}(elem) +  0x9e3779b9 + (seed <<  6) + (seed >>  2);
        }
        return seed;
    }
};

namespace Brush{


// TODO: separate declaration from implementation
// TODO: move fitness to eval folder
// TODO make a better use of this (in selection, when fitting, etc)  (actually i need to start using it)
struct Fitness {

    float loss;     ///< aggregate loss score
    float loss_v;   ///< aggregate validation loss score

    // TODO: maybe this should be all part of fitness, and individual should have only the fitness, program, and error (and objectives)
    size_t complexity;
    size_t size;
    size_t depth;


    // these can be different depending on the island the individual is
    unsigned int dcounter;  ///< number of individuals this dominates
    vector<unsigned int> dominated; ///< individual indices this dominates
    unsigned int rank;             ///< pareto front rank
    float crowding_dist;   ///< crowding distance on the Pareto front

    void set_dominated(vector<unsigned int>& dom){ dominated=dom; };
    vector<unsigned int> get_dominated() const { return dominated; };

    void set_loss(float f){ loss=f; };
    float get_loss() const { return loss; };

    void set_loss_v(float f_v){ loss_v=f_v; };
    float get_loss_v() const { return loss_v; };

    void set_dcounter(unsigned int d){ dcounter=d; };
    unsigned int get_dcounter() const { return dcounter; };

    void set_rank(unsigned r){ rank=r; };
    size_t get_rank() const { return rank; };

    void set_crowding_dist(float cd){ crowding_dist=cd; };
    float get_crowding_dist() const { return crowding_dist; };

    vector<float> values;
    vector<float> weights;

    // TODO: fitness could have a function size()

    // weighted values
    vector<float> wvalues;

    // Constructor with initializer list for weights
    Fitness(const vector<float>& w={}) : values(), wvalues(), weights(w) {
        dcounter = 0;
        set_rank(0);
        set_crowding_dist(0);
        dominated.resize(0);
    }
    
    // Hash function
    size_t hash() const {
        std::size_t h = std::hash<vector<float>>{}(wvalues);
        return h;
    }

    void set_weights(vector<float>& w) {
        weights = w;
    }
    vector<float> get_weights() const {
        return weights;
    }
    vector<float> get_values() const {
        return values;
    }
    vector<float> get_wvalues() const {
        return wvalues;
    }

    // TODO: debug size, it is giving weird values
    // Method to set values
    void set_values(vector<float>& v) {
        if (v.size() != weights.size()) {
            throw std::length_error("Assigned values have not the same length than current values");
        }
        // fmt::print("updated values\n");

        values.resize(0);
        for (const auto& element : v) {
            values.push_back(element);
        }

        wvalues.resize(weights.size());

        // Perform element-wise multiplication
        std::transform(v.begin(), v.end(), 
                       weights.begin(), wvalues.begin(),
                        [](double a, double b) {
                            return a * b;
                        });
    }

    // Method to clear values
    void clearValues() {
        wvalues.clear();
    }

    bool valid() const {
        return !wvalues.empty();
    }

    // Equality comparison
    bool operator==(const Fitness& other) const {
        return wvalues == other.wvalues;
    }

    // Inequality comparison
    bool operator!=(const Fitness& other) const {
        return !(*this == other);
    }

    // Less than comparison
    bool operator<(const Fitness& other) const {
        return std::lexicographical_compare(wvalues.begin(), wvalues.end(),
                                            other.wvalues.begin(), other.wvalues.end());
    }

    // Greater than comparison
    bool operator>(const Fitness& other) const {
        return other < *this;
    }

    // Less than or equal to comparison
    bool operator<=(const Fitness& other) const {
        return !(other < *this);
    }

    // Greater than or equal to comparison
    bool operator>=(const Fitness& other) const {
        return !(*this < other);
    }

    // String representation
    std::string toString() const {
        if (valid()) {
            return "TODO: implement string representation"; //std::to_string(wvalues);
        } else {
            return "Tuple()";
        }
    }

    // Representation for debugging
    std::string repr() const {
        return "Fitness(TODO: implement string representation)";
    }


    /// set obj vector given a string of objective names
    int dominates(const Fitness& b) const;
};

void to_json(json &j, const Fitness &f);
void from_json(const json &j, Fitness& f);

namespace Pop{
    
template<ProgramType T> 
class Individual{
public: // TODO: make these private (and work with nlohman json)
    Program<T> program; ///< executable data structure

    // store just info that we dont have a getter. size, depth, complexity: they can all be obtained with program.<function here>

    VectorXf error;     ///< training error (used in lexicase selectors)

    Fitness fitness;     ///< aggregate fitness score

    vector<string> objectives; ///< objectives for use with Pareto selection
       
    Individual()
    {
        // TODO: default value for fitness
        // the fitness is used in evolutionary functions
        // fitness = -1;
        
        // loss is the aggregation of error vector, and can be user sppecified
        // loss = -1;
        // loss_v = -1;

        // complexity=-1;
        // size=-1;
        // depth=-1;

        // dcounter=-1;
        // rank=-1;
        // crowding_dist = -1;

        objectives = {"error", "complexity"}; 
    };

    Individual(Program<T>& prg) : Individual() { program = prg; };

    // TODO: clone? maybe a constructor that takes another individual as arg and copies  everything

    void init(SearchSpace& ss, const Parameters& params)
    {
        program = ss.make_program<Program<T>>(params, 0, 0);

        // If different from zero, then the program is created with a fixed depth and size.
        // If zero, it samples the value
        // program = SS.make_program<T>(params, params.max_depth, params.max_size);
    };

    // fitness, objetives, complexity, etc. TODO: create intermediate  functions to  interact  with fitness and program?
    void fit(Dataset& data) {
        program.fit(data);
        
    };
    auto predict(Dataset& data) { return program.predict(data); };

    // TODO: predict proba and classification related methods.
    // TODO: This class should also have its own cpp wrapper. Update it into the deap api (the idea is that the user is still able to prototype with brush, I dont think we should disable that feature)

    // just getters (TODO: use the attributes )
    string get_model() const { return program.get_model(); };
    size_t get_size() const { return program.size(); };
    size_t get_depth() const { return program.depth(); };
    size_t get_complexity() const { return program.complexity(); };
    Program<T>& get_program() { return program; };

    void set_fitness(Fitness &f) { fitness=f; };
    Fitness& get_fitness() { return fitness; };

    // TODO: USE setters and getters intead of accessing it directly
    // template<ProgramType T>
    // void Individual<T>::set_objectives(const vector<string>& objectives)

    // TODO:  fix   to use these with fitness instead of with individual
    // unsigned int dcounter;  ///< number of individuals this dominates
    // vector<unsigned int> dominated; ///< individual indices this dominates
    
    // unsigned int rank;             ///< pareto front rank
    // float crowding_dist;   ///< crowding distance on the Pareto front


    // Static map for weights associated with strings
    // TODO: weights for different values. loss should be calculated duing runtime, based on the metric
    inline static std::map<std::string, float> weightsMap = []() {
        std::map<std::string, float> map = {
            {"complexity", -1.0},
            {"size", -1.0}
            // Add more key-value pairs as needed
        };
        // example on how to have weight based on templated class
        map["error"] = (T == Brush::ProgramType::Regressor) ?  -1.0 : -1.0;

        return map;
    }();

    vector<string> get_objectives() const { return objectives; };
    void set_objectives(vector<string> objs){
        objectives=objs;
        
        vector<float> weights;
        weights.resize(0);
        for (const auto& obj : objectives) {
            auto it = weightsMap.find(obj);
            if (it != weightsMap.end()) {
                weights.push_back(it->second);
            } else {
                // TODO: throw error here, unknown objective
                std::cout << obj << " not found in the weight map." << std::endl;
            }
        }

        fitness.set_weights(weights);
    };
};


// TODO: rename (something better (more meaningful) than p)
// serialization for Individual
template<ProgramType T>
void to_json(json &j, const Individual<T> &p)
{
    j = json{
        // TODO: jsonify fitness struct, and new possible obj functions
        {"program", p.program},
        {"fitness", p.fitness},
        // {"loss", p.loss},
        // {"loss_v", p.loss_v},
        // {"complexity", p.complexity},
        // {"size", p.size},
        // {"depth", p.depth},
        // {"rank", p.rank},
        // {"crowding_dist", p.crowding_dist},
        {"objectives", p.objectives}
    }; 
}

template<ProgramType T>
void from_json(const json &j, Individual<T>& p)
{// TODO: figure  out if this works  with private attributes and try to actually make them private (and use getters and setters)
    j.at("program").get_to( p.program );
    j.at("fitness").get_to( p.fitness );
    // j.at("loss").get_to( p.loss );
    // j.at("loss_v").get_to( p.loss_v );
    // j.at("complexity").get_to( p.complexity );
    // j.at("size").get_to( p.size );
    // j.at("depth").get_to( p.depth );
    // j.at("rank").get_to( p.rank );
    // j.at("crowding_dist").get_to( p.crowding_dist );
    j.at("objectives").get_to( p.objectives );
}
} // Pop
} // Brush

#endif
