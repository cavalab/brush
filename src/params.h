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
    // TODO: setters and getters for all parameters? (and do checks in setters?). Also make them private, and use the getters and setters in the code
    
    // settings
    int random_state; // TODO: constructor should set the global rng to random_state (if given, otherwise just let it work normally)
    //int verbosity = 0; // TODO: implement log and verbosity    

    // TODO: every parameter should have a default value
    // Evolutionary stuff
    string mode="regression"; 

    unsigned int current_gen = 1;

    int pop_size           = 100;
    int gens               = 1000;      
    unsigned int max_depth = 6; // TODO: make all tests be based on these values for max depth and size
    unsigned int max_size  = 50;
    vector<string> objectives{"error","complexity"}; // error should be generic and deducted based on mode
    string sel  = "lexicase"; //selection method
    string surv = "nsga2"; //survival method
    std::unordered_map<string, float> functions;
    int num_islands=5;

    // variation
    std::map<std::string, float> mutation_probs = {
        {"point", 0.167},
        {"insert", 0.167},
        {"delete", 0.167},
        {"subtree", 0.167},
        {"toggle_weight_on", 0.167},
        {"toggle_weight_off", 0.167}
    };

    float cx_prob=0.2;         ///< cross rate for variation
    float mig_prob = 0.05;
    
    string scorer_="mse";   ///< actual loss function used, determined by error

    // for classification (TODO: should I have these, or they could be just dataset arguments (except the ones needed to use in dataset constructor))
    bool classification;
    unsigned int n_classes;   ///< number of classes for classification 

    // TODO: set these values when creating the parameters in python side
    vector<int> classes;      ///< class labels
    vector<float> class_weights;  ///< weights for each class
    vector<float> sample_weights; ///< weights for each sample 
    
    // for dataset. TODO: make it work
    bool shuffle = true;             ///< option to shuffle the data
    float split = 0.75;              ///< fraction of data to use for training
    vector<string> feature_names; ///< names of features
    float batch_size = 0.0;
    bool use_batch = false; ///< whether to use mini batch for training

    int n_jobs = 1; // -1; ///< number of parallel jobs -1 use all threads; 0 use same as number of islands; positive number specify the amouut of threads

    Parameters(){}; 
    ~Parameters(){};

    void set_pop_size(int new_pop_size){ pop_size = new_pop_size; };
    int get_pop_size(){ return pop_size; };

    void set_gens(int new_gens){ gens = new_gens; };
    int get_gens(){ return gens; };

    void set_current_gen(unsigned int gen){ current_gen = gen; };
    unsigned int get_current_gen(){ return current_gen; };

    void set_num_islands(int new_num_islands){ num_islands = new_num_islands; };
    int get_num_islands(){ return num_islands; };

    void set_max_depth(unsigned new_max_depth){ max_depth = new_max_depth; };
    unsigned get_max_depth(){ return max_depth; };

    void set_max_size(unsigned new_max_size){ max_size = new_max_size; };
    unsigned get_max_size(){ return max_size; };

    void set_objectives(vector<string> new_objectives){ objectives = new_objectives; };
    vector<string> get_objectives(){ return objectives; };

    void set_sel(string new_sel){ sel = new_sel; };
    string get_sel(){ return sel; };

    void set_surv(string new_surv){ surv = new_surv; };
    string get_surv(){ return surv; };

    void set_cx_prob(float new_cx_prob){ cx_prob = new_cx_prob; };
    float get_cx_prob(){ return cx_prob; };

    void set_mig_prob(float new_mig_prob){ mig_prob = new_mig_prob; };
    float get_mig_prob(){ return mig_prob; };

    void set_classification(bool c){ classification = c; };
    bool get_classification(){ return classification; };

    void set_n_classes(unsigned int new_n_classes){ n_classes = new_n_classes; };
    unsigned int get_n_classes(){ return n_classes; };

    //TODO: unify unordered or ordered
    void set_mutation_probs(std::map<std::string, float> new_mutation_probs){ mutation_probs = new_mutation_probs; };
    std::map<std::string, float> get_mutation_probs(){ return mutation_probs; };

    void set_functions(std::unordered_map<std::string, float> new_functions){ functions = new_functions; };
    std::unordered_map<std::string, float> get_functions(){ return functions; };
};

// Global (deprecated) params
extern ns::json PARAMS; 
void set_params(const ns::json& j);
ns::json get_params();

} // Brush

#endif
