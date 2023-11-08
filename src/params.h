/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef PARAMS_H
#define PARAMS_H

#include "init.h"

namespace ns = nlohmann;
namespace Brush
{

struct Parameters
{
public:
    // TODO: setters and getters for all parameters? (and do checks in setters?)

    // settings
    int random_state; // TODO: constructor should set the global rng to random_state (if given, otherwise just let it work normally)
    //int verbosity = 0; // TODO: implement log and verbosity    

    // TODO: every parameter should have a default value
    // TODO: python wrapper should have getters and setters for all this stuff
    // Evolutionary stuff
    string mode="regression"; 

    unsigned int current_gen = 1;

    int pop_size = 100;
    int gens = 100;      
    unsigned int max_depth=10;
    unsigned int max_size=100;
    vector<string> objectives{"error","complexity"}; // error should be generic and deducted based on mode
    string sel = "nsga2"; //selection method
    string surv = "nsga2"; //survival method
    vector<string> functions;
    int num_islands=5;

    // variation
    std::map<std::string, float> mutation_probs;
    float cx_prob=0.2;         ///< cross rate for variation
    float mig_prob = 0.05;
    
    string scorer_;   ///< actual loss function used, determined by error

    // for classification (TODO: should I have these, or they could be just dataset arguments (except the ones needed to use in dataset constructor))
    unsigned int n_classes;   ///< number of classes for classification 
    vector<int> classes;      ///< class labels
    vector<float> class_weights;  ///< weights for each class
    vector<float> sample_weights; ///< weights for each sample 
    
    // for dataset
    bool shuffle = true;             ///< option to shuffle the data
    float split = 0.75;              ///< fraction of data to use for training
    vector<string> feature_names; ///< names of features
    float batch_size = 0.0;
    bool use_batch = false; ///< whether to use mini batch for training

    int n_jobs = 1; ///< number of parallel jobs (TODO if -1, equals the number of islands?)

    Parameters(){}; 
    ~Parameters(){};
};

// Global (deprecated) params
extern ns::json PARAMS; 
void set_params(const ns::json& j);
ns::json get_params();

} // Brush

#endif
