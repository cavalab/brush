#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "../program/program.h"
#include "fitness.h"

#include <functional>

using namespace nlohmann;

namespace Brush{
namespace Pop{

template<ProgramType T> 
class Individual{
public: // TODO: make these private (and work with nlohman json)
    Program<T> program; ///< executable data structure

    // store just info that we dont have a getter. size, depth, complexity: they can all be obtained with program.<function here>

    // error is the aggregation of error vector, and can be user sppecified
    
    // this flag is used to avoid re-fitting an individual. the program is_fitted_ flag is used to perform checks (like in predict with weights). They are two different things and I think I;ll keep this way (individual is just a container to keep program and fitness together) 
    bool is_fitted_ = false;

    // archive utility (and also keep track of evolution) (this is meaningful only
    // if variation is done using the vary() function)
    unsigned id;                 ///< tracking id
    vector<unsigned> parent_id;  ///< ids of parents
    
    // storing what changed in relation to parent inside variation
    string variation = "born"; // spontanegous generation (born), crossover, or which type of mutation

    VectorXf error;     ///< training error (used in lexicase selectors)

    Fitness fitness;     ///< aggregate fitness score

    vector<string> objectives; ///< objectives for use with Pareto selection
       

    Individual()
    {
        objectives = {"error", "complexity"}; 
        id = 0; // unsigned
    };

    Individual(Program<T>& prg) : Individual() { program = prg; };

    void init(SearchSpace& ss, const Parameters& params)
    {
        program = ss.make_program<Program<T>>(params, 0, 0);

        // If different from zero, then the program is created with a fixed depth and size.
        // If zero, it samples the value
        // program = SS.make_program<T>(params, params.max_depth, params.max_size);
    };

    // TODO: replace occurences of program.fit with these (also predict and predict_proba)
    Individual<T> &fit(const Dataset& data) {
        program.fit(data);
        this->is_fitted_ = true;
        return *this;
    };
    Individual<T> &fit(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)
    {
        Dataset d(X,y);
        return fit(d);
    };

    auto predict(const Dataset& data) { return program.predict(data); };
    auto predict(const Ref<const ArrayXXf>& X)
    {
        Dataset d(X);
        return predict(d);
    };

    template <ProgramType P = T>
        requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
    auto predict_proba(const Dataset &d) { return program.predict_proba(d); };
    template <ProgramType P = T>
        requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
    auto predict_proba(const Ref<const ArrayXXf>& X) 
    {
        Dataset d(X);
        return predict_proba(d);
    };

    // just getters
    bool get_is_fitted() const { return this->is_fitted_; };
    unsigned int get_size() const { return program.size(); };
    unsigned int get_depth() const { return program.depth(); };
    unsigned int get_complexity() const { return program.complexity(); };
    Program<T>& get_program() { return program; };
    
    string get_model(string fmt="compact", bool pretty=false) {
        return program.get_model(fmt, pretty); };
    string get_dot_model(string extras="") {
        return program.get_dot_model(extras); };

    void set_fitness(Fitness &f) { fitness=f; };
    Fitness& get_fitness() { return fitness; };

    void set_variation(string v) { variation=v; };
    string get_variatiion() { return variation; };
    
    void set_id(unsigned i){id = i;};
    void set_parents(const vector<Individual<T>>& parents){
        parent_id.clear();
        for (const auto& p : parents)
            parent_id.push_back(p.id);
    };     /// set parent ids using parents  
    void set_parents(const vector<unsigned>& parents){ parent_id = parents; };     /// set parent ids using id values 

    // TODO: USE setters and getters intead of accessing it directly
    // template<ProgramType T>
    // void Individual<T>::set_objectives(const vector<string>& objectives)

    // Static map for weights associated with strings.
    // this will determine each fitness metric to be a min/max problem.
    // generic error metric: by default log and multi_log if it is a
    // classification problem, and MSE if it is a regression (so its always
    // a minimization by default, thus "error" has weight -1.0)
    inline static std::map<std::string, float> weightsMap = {
        {"complexity",              -1.0},
        {"size",                    -1.0},
        {"mse",                     -1.0},
        {"log",                     -1.0},
        {"multi_log",               -1.0},
        {"average_precision_score", +1.0},
        {"accuracy",                +1.0},
        {"error",                   -1.0}
    };

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
                throw std::runtime_error(
                    "Unknown metric used as fitness. Value was " + obj);
            }
        }

        fitness.set_weights(weights);
    };
};


// serialization for Individual
template<ProgramType T>
void to_json(json &j, const Individual<T> &p)
{
    j = json{
        {"program", p.program},
        {"fitness", p.fitness},
        {"id", p.id},
        {"parent_id", p.parent_id},
        {"objectives", p.objectives}
    }; 
}

template<ProgramType T>
void from_json(const json &j, Individual<T>& p)
{// TODO: figure  out if this works  with private attributes and try to actually make them private (and use getters and setters)
    j.at("program").get_to( p.program );
    j.at("fitness").get_to( p.fitness );
    j.at("id").get_to( p.id );
    j.at("parent_id").get_to( p.parent_id );
    j.at("objectives").get_to( p.objectives );
}
} // Pop
} // Brush

#endif
