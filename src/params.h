/* Brush
copyright 2024 William La Cava
license: GNU/GPL v3
*/

#ifndef PARAMS_H
#define PARAMS_H

#include "util/logger.h"
#include "util/utils.h"

namespace ns = nlohmann;

namespace Brush
{

using namespace Util;

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

    unsigned int max_depth = 5;
    unsigned int max_size  = 50;

    vector<string> objectives{"scorer","linear_complexity"}; // scorer should be generic and deducted based on mode
    string bandit = "thompson"; // TODO: should I rename dummy? 
    string sel  = "lexicase"; //selection method
    string surv = "nsga2"; //survival method
    std::unordered_map<string, float> functions;
    int num_islands=1;

    // Different simplification strategies
    bool constants_simplification=true;
    bool inexact_simplification=true;

    // population initialization
    bool start_from_decision_trees=false;

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

    vector<int>   classes = vector<int>();          ///< class labels
    vector<float> class_weights = vector<float>();  ///< weights for each class
    vector<float> sample_weights = vector<float>(); ///< weights for each sample 
    string        class_weights_type = "unbalanced"; // user_defined, unbalanced, support

    // for creating dataset from X and y in Engine<T>::fit. Ignored if 
    // the uses uses an dataset
    bool classification = false;
    unsigned int n_classes = 0;

    // validation partition
    bool shuffle_split = false;
    float validation_size = 0.2;
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

    // TODO: improve vary_and_update to have island working in parallel 
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
        
        // properly replace scorer with the specified scorer
        vector<string> aux_objectives(0);

        for (auto& objective : objectives) {
            if (objective.compare("scorer")==0)
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

    void set_classification(bool c){ classification = c; };
    bool get_classification(){ return classification; };

    void set_shuffle_split(bool shuff){ shuffle_split = shuff; };
    bool get_shuffle_split(){ return shuffle_split; };

    void set_constants_simplification(bool cs){ constants_simplification = cs; };
    bool get_constants_simplification(){ return constants_simplification; };
    
    void set_inexact_simplification(bool is){ inexact_simplification = is; };
    bool get_inexact_simplification(){ return inexact_simplification; };

    void set_weights_init(bool init){ weights_init = init; };
    bool get_weights_init(){ return weights_init; };

    void set_n_classes(const ArrayXf& y){
        if (classification)
        {
            vector<int> uc = unique( ArrayXi(y.cast<int>()) );

            if (int(uc.at(0)) != 0)
                HANDLE_ERROR_THROW("Class labels must start at 0");

            vector<int> cont_classes(uc.size());
            iota(cont_classes.begin(), cont_classes.end(), 0);
            for (int i = 0; i < cont_classes.size(); ++i)
            {
                if ( int(uc.at(i)) != cont_classes.at(i))
                    HANDLE_ERROR_THROW("Class labels must be contiguous");
            }
            n_classes = uc.size();
            // classes   = uc;
        }
    };
    void set_class_weights(const vector<float>& weights){
        if (weights.size() != n_classes)
            HANDLE_ERROR_THROW("Length of class_weights does not match expected number of classes");

        class_weights.clear();
        for (unsigned int i = 0; i < n_classes; ++i) {
            class_weights.push_back(weights[i]);
        }
    };

    void set_sample_weights(const ArrayXf& y){
        sample_weights.resize(0);

        // one if for each case, so the default is unbalanced or user defined
        if (class_weights_type == "support")
        { // ignores everything and calculate the weights here.
            class_weights.resize(n_classes); // set_n_classes must be called first
            for (unsigned i = 0; i < n_classes; ++i){
                // weighting by support 
                int support = (y.cast<int>().array() == i).count();

                if (support==0)
                    class_weights.at(i) = 0.0;
                else
                    class_weights.at(i) = float(y.size()) / float(n_classes * support);
            }
        } // else it is either unbalanced or user_defined

        if (!class_weights.empty())
            for (unsigned i = 0; i < y.size(); ++i)
                sample_weights.push_back(class_weights.at(int(y(i))));
    };

    unsigned int get_n_classes(){ return n_classes; };
    vector<float> get_class_weights(){ return class_weights; };
    vector<float> get_sample_weights(){ return sample_weights; };

    string get_class_weights_type(){ return class_weights_type; };
    void set_class_weights_type(string cwt){ class_weights_type = cwt; };

    bool get_start_from_decision_trees(){ return start_from_decision_trees; };
    void set_start_from_decision_trees(bool start_dt){ start_from_decision_trees = start_dt; };

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
    random_state,
    verbosity,

    mode,

    current_gen,

    pop_size,
    max_gens,
    max_stall,
    max_time,

    max_depth,
    max_size,

    objectives,
    bandit,
    sel,
    surv,
    functions,
    num_islands,
    
    constants_simplification,
    inexact_simplification,

    mutation_probs,
    cx_prob,
    mig_prob,

    scorer,

    classes, // TODO: get rid of this parameter? for some reason, when i remove it (or set it to any value) the load population starts to fail with regression
    class_weights,
    sample_weights,

    classification,
    n_classes,

    start_from_decision_trees,

    shuffle_split,
    validation_size,
    feature_names,
    feature_types,
    batch_size,
    weights_init,

    load_population,
    save_population,

    logfile,
    
    n_jobs
);

} // Brush

#endif
