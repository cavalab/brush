/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef PARAMS_H
#define PARAMS_H

#include "init.h"
#include "util/logger.h"

namespace ns = nlohmann;

namespace Brush
{

struct Parameters
{
public:
    // by default, the rng generator will use any random seed if random_state is zero
    int random_state = 0;
    int verbosity = 0; 

    // Evolutionary algorithm settings
    string mode="regression"; 

    unsigned int current_gen = 1;

    // termination criteria
    int pop_size  = 100;
    int max_gens  = 100;
    int max_stall = 0;
    int max_time  = -1;

    unsigned int max_depth = 6;
    unsigned int max_size  = 50;

    vector<string> objectives{"error","complexity"}; // error should be generic and deducted based on mode
    string bandit = "dynamic_thompson"; // TODO: should I rename dummy? 
    string sel  = "lexicase"; //selection method
    string surv = "nsga2"; //survival method
    std::unordered_map<string, float> functions;
    int num_islands=5;

    // if we should save pareto front of the entire evolution (use_arch=true)
    // or just the final population (use_arch=false)
    bool use_arch=false;
    bool val_from_arch=true;

    // variation
    std::map<std::string, float> mutation_probs = {
        {"point", 0.167},
        {"insert", 0.167},
        {"delete", 0.167},
        {"subtree", 0.167},
        {"toggle_weight_on", 0.167},
        {"toggle_weight_off", 0.167}
    };

    float cx_prob=0.2;  ///< cross rate for variation
    float mig_prob = 0.05;
    
    string scorer="mse";   ///< actual loss function used, determined by error

    vector<int>   classes;        ///< class labels
    vector<float> class_weights;  ///< weights for each class
    vector<float> sample_weights; ///< weights for each sample 
    
    // for creating dataset from X and y in Engine<T>::fit. Ignored if 
    // the uses uses an dataset
    bool classification;
    unsigned int n_classes;

    // validation partition
    bool shuffle_split = false;
    float validation_size = 0.75;
    vector<string> feature_names = {};
    vector<string> feature_types = {};
    float batch_size = 0.0;
    bool weights_init=true;

    string load_population = "";
    string save_population = "";

    string logfile = "";

    int n_jobs = 1; ///< number of parallel jobs -1 use all threads; 0 use same as number of islands; positive number specify the amouut of threads

    Parameters(){}; 
    ~Parameters(){};
    
    // TODO: use logger to log information. Make getters const  
    void set_verbosity(int new_verbosity){ Brush::Util::logger.set_log_level(new_verbosity);
                                           verbosity = new_verbosity; };
    int get_verbosity(){ return verbosity; };

    void set_random_state(int new_random_state){random_state = new_random_state; };
    int get_random_state(){ return random_state; };

    void set_pop_size(int new_pop_size){ pop_size = new_pop_size; };
    int get_pop_size(){ return pop_size; };

    void set_max_gens(int new_max_gens){ max_gens = new_max_gens; };
    int get_max_gens(){ return max_gens; };
    
    void set_bandit(string new_bandit){ bandit = new_bandit; };
    string get_bandit(){ return bandit; };

    void set_max_stall(int new_max_stall){ max_stall = new_max_stall; };
    int get_max_stall(){ return max_stall; };

    void set_max_time(int new_max_time){ max_time = new_max_time; };
    int get_max_time(){ return max_time; };
    
    void set_scorer(string new_scorer){ scorer = new_scorer; };
    string get_scorer(){ return scorer; };

    void set_load_population(string new_load_population){ load_population = new_load_population; };
    string get_load_population(){ return load_population; };
    
    void set_save_population(string new_save_population){ save_population = new_save_population; };
    string get_save_population(){ return save_population; };

    string get_logfile(){ return logfile; };
    void set_logfile(string s){ logfile=s; };
    
    void set_current_gen(unsigned int gen){ current_gen = gen; };
    unsigned int get_current_gen(){ return current_gen; };

    void set_num_islands(int new_num_islands){ num_islands = new_num_islands; };
    int get_num_islands(){ return num_islands; };

    void set_max_depth(unsigned new_max_depth){ max_depth = new_max_depth; };
    unsigned get_max_depth() const { return max_depth; };

    void set_n_jobs(int new_n_jobs){ n_jobs = new_n_jobs; };
    int get_n_jobs(){ return n_jobs; };

    void set_max_size(unsigned new_max_size){ max_size = new_max_size; };
    unsigned get_max_size() const { return max_size; };

    void set_objectives(vector<string> new_objectives){ objectives = new_objectives; };
    vector<string> get_objectives() const {
        // return objectives;
        
        // properly replace error with the specified scorer
        vector<string> aux_objectives(0);

        for (auto& objective : objectives) {
            if (objective.compare("error")==0)
                aux_objectives.push_back(scorer);
            else
                aux_objectives.push_back(objective);
        }

        return aux_objectives;
    };

    void set_sel(string new_sel){ sel = new_sel; };
    string get_sel(){ return sel; };

    void set_surv(string new_surv){ surv = new_surv; };
    string get_surv(){ return surv; };

    void set_cx_prob(float new_cx_prob){ cx_prob = new_cx_prob; };
    float get_cx_prob(){ return cx_prob; };

    void set_mig_prob(float new_mig_prob){ mig_prob = new_mig_prob; };
    float get_mig_prob(){ return mig_prob; };

    void set_use_arch(bool new_use_arch){ use_arch = new_use_arch; };
    bool get_use_arch(){ return use_arch; };

    void set_val_from_arch(bool new_val_from_arch){ val_from_arch = new_val_from_arch; };
    bool get_val_from_arch(){ return val_from_arch; };

    void set_classification(bool c){ classification = c; };
    bool get_classification(){ return classification; };

    void set_shuffle_split(bool shuff){ shuffle_split = shuff; };
    bool get_shuffle_split(){ return shuffle_split; };

    void set_weights_init(bool init){ weights_init = init; };
    bool get_weights_init(){ return weights_init; };

    void set_n_classes(unsigned int new_n_classes){ n_classes = new_n_classes; };
    unsigned int get_n_classes(){ return n_classes; };

    void set_validation_size(float s){ validation_size = s; };
    float get_validation_size(){ return validation_size; };

    void set_feature_names(vector<string> vn){ feature_names = vn; };
    vector<string> get_feature_names(){ return feature_names; };

    void set_feature_types(vector<string> ft){ feature_types = ft; };
    vector<string> get_feature_types(){ return feature_types; };

    void set_batch_size(float c){ batch_size = c; };
    float get_batch_size(){ return batch_size; };

    void set_mutation_probs(std::map<std::string, float> new_mutation_probs){ mutation_probs = new_mutation_probs; };
    std::map<std::string, float> get_mutation_probs(){ return mutation_probs; };

    void set_functions(std::unordered_map<std::string, float> new_functions){ functions = new_functions; };
    std::unordered_map<std::string, float> get_functions(){ return functions; };
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Parameters,
    verbosity,
    random_state,
    pop_size,
    max_gens,
    max_stall,
    max_time,
    scorer,
    load_population,
    save_population,
    logfile,
    current_gen,
    num_islands,
    max_depth,
    n_jobs,
    max_size,
    objectives,
    sel,
    surv,
    cx_prob,
    mig_prob,
    classification,
    n_classes,
    validation_size,
    feature_names,
    batch_size,
    mutation_probs,
    functions
);

} // Brush

#endif
