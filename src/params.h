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
private:
    // settings
    int random_state; // TODO: constructor should set the global rng to random_state (if given, otherwise just let it work normally)
    int verbosity = 0;    

    // TODO: python wrapper should have getters and setters for all this stuff
    // Evolutionary stuff
    string mode="regression"; 
    int pop_size = 100;
    int gens = 100;      
    unsigned int max_depth = 10;
    unsigned int max_size=100;
    vector<string> objectives{"error","complexity"}; // error should be generic and deducted based on mode
    float cx_prob;         ///< cross rate for variation
    float mutation_probs;
    int num_islands=5;
    float mig_prob = 0.05;
    vector<string> functions;
    string scorer_;   ///< actual loss function used, determined by error

    // for classification 
    unsigned int n_classes;   ///< number of classes for classification 
    vector<int> classes;      ///< class labels
    vector<float> class_weights;  ///< weights for each class
    vector<float> sample_weights; ///< weights for each sample 
    
    // from dataset
    bool shuffle = true;             ///< option to shuffle the data
    float split = 0.75;              ///< fraction of data to use for training
    vector<string> feature_names; ///< names of features
    float batch_size = 0.0;
    bool use_batch = false; ///< whether to use mini batch for training

    int n_jobs = 1; ///< number of parallel jobs
public:
    Parameters() {}; 
    ~Parameters(){};

    // TODO: getters and setters

    void init(const MatrixXf& X, const VectorXf& y);
};

// Global (deprecated) params
extern ns::json PARAMS; 
void set_params(const ns::json& j);
ns::json get_params();

} // Brush

#endif
