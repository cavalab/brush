/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef CBrush_H
#define CBrush_H

#include "init.h"
#include "population.h"
#include "params.h"
#include "./eval/evaluation.h"
#include "selection/selection.h"
#include "./util/rnd.h"
#include "taskflow/taskflow.hpp"

// TODO: improve the includes (why does this lines below does not work?)
// #include "variation.h"
// #include "selection.h"


namespace Brush
{

using namespace Pop;
using namespace Sel;
using namespace Eval;

// using namespace variation;

template <ProgramType T>
class CBrush{ // TODO: rename it to BrushEstimator
public:
    CBrush()
    : params(Parameters())
    , ss(SearchSpace())
    , variator(Variation<T>(params, ss))
    {};

    ~CBrush(){};
    void init();

    //getters and setters for GA configuration ---------------------------------
    /// set flag indicating whether fit has been called
    inline void set_is_fitted(bool f){is_fitted=f;}
    inline bool get_is_fitted(){return is_fitted;}

    // TODO: WRAPPER SHOULD SET ALL THESE (by changing the inner parameter instance)

    void set_pop_size(int pop_size){ params.pop_size = pop_size; };
    int get_pop_size(){ return params.pop_size; };
    
    void set_gens(int gens){ params.gens = gens; };
    int get_gens(){ return params.gens; };
                
    void set_max_depth(unsigned int max_depth){ params.max_depth = max_depth; };
    int get_max_depth(){ return params.max_depth; };

    void set_max_size(unsigned int max_size){ params.max_size = max_size; };
    int get_max_size(){ return params.max_size; };
    
    void set_mode(string mode) { params.mode = mode; };
    string get_mode(){ return params.mode; };

    void set_selection(string sel){ params.sel = sel; };
    string get_selection(){ return params.sel; };

    void set_survival(string surv){ params.surv = surv; };
    string get_survival(){ return params.surv; };
                    
    void set_num_islands(int n_islands){ params.num_islands = n_islands; };
    int get_num_islands(){ return params.num_islands; };
             
    void set_objectives(const vector<string>& obj){params.objectives = obj; };
    auto get_objectives(){return params.objectives; };  
         
    void set_random_state(int random_state) {
        params.random_state = random_state;
        r.set_seed(params.random_state);
    };
    int get_random_state() { return params.random_state; };
    
    void set_mig_prob(float mig_prob){ params.mig_prob = mig_prob;};
    float get_mig_prob(){ return params.mig_prob; };
    
    void set_cross_prob(float cross_prob){ params.cx_prob = cross_prob;};
    float get_cross_prob(){ return params.cx_prob; };
    
    // sets available functions based on comma-separated list.
    void set_functions(const vector<string>& fns){ params.functions = fns; };
    unordered_map<string, float> get_functions(){ return params.functions; };
                
    void set_mutation_probs(std::map<std::string, float> mutation_probs){ params.mutation_probs = mutation_probs;};
    std::map<std::string, float> get_mutation_probs(){ return params.mutation_probs; };
    
    //TODO
    ///return fraction of data to use for training
    // float get_split();
    // /// set train fraction of dataset
    // void set_split(float sp);

    // TODO
    // int get_batch_size(){return params.bp.batch_size;};
    // void set_batch_size(int bs);
      
    // TODO
    ///set number of threads (and use them in taskflow)
    // void set_n_jobs(unsigned t);
    // int get_n_jobs(){return omp_get_num_threads();};
    
    ///set flag to use batch for training
    // void set_use_batch();
    
    // TODO getters and setters for the best solution found after evolution
    // predict, transform, predict_proba, etc.
    // get statistics
    // load and save best individuals
    // logger, save to file
    // execution archive
    // score functions
    // fit methods (this will run the evolution)

    /// updates best score
    bool update_best(const Dataset& data, bool val=false);    
    
    /// train a model. TODO: take arguments needed to build the dataset. once we have it, go through params to set global options and use them
    void fit(MatrixXf& X);
    void fit(MatrixXf& X, VectorXf& y);
    
    bool is_fitted; ///< keeps track of whether fit was called.

    void run_generation(unsigned int g, Dataset &data);
private:
    Parameters params;  ///< hyperparameters of brush 
    SearchSpace ss;

    Population<T> pop;       	///< population of programs
    Selection<T>  selector;   ///< selection algorithm
    Evaluation<T> evaluator;  ///< evaluation code
    Variation<T>  variator;  	///< variation operators
    Selection<T>  survivor;   ///< survival algorithm
    
    // TODO: MISSING CLASSES: timer, archive, logger
    float best_loss;
    int best_complexity;
    Individual<T> best_ind;

    // calculate/print stats
};

} // Brush

#endif