/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef CBrush_H
#define CBrush_H

#include "init.h"
#include "params.h"
#include "selection/selection.h"
#include "population.h"
#include "taskflow/taskflow.hpp"

// TODO: improve the includes (why does this lines below does not work?)
// #include "variation.h"
// #include "selection.h"

// using namespace selection;
// using namespace variation;

namespace Brush
{

class CBrush{
public:
    CBrush(){};         // TODO: constructor should create a new parameters and use it in every other stuff
    ~CBrush(){};
    void init();

    //getters and setters for GA configuration ---------------------------------
    /// set flag indicating whether fit has been called
    inline void set_is_fitted(bool f){is_fitted=f;}
    inline bool get_is_fitted(){return is_fitted;}

    /// set size of population 
    void set_pop_size(int pop_size);
    /// return population size
    int get_pop_size();
    
    /// set size of max generations              
    void set_gens(int gens);
    ///return size of max generations
    int get_gens();
                
    /// set EProblemType for shogun              
    void set_classification(bool classification);
    ///return type of classification flag set
    bool get_classification();
          
    /// set selection method              
    void set_selection(string sel);
    string get_selection();
                
    /// set survivability              
    void set_survival(string surv);
    string get_survival();
                
    ///return cross rate for variation
    float get_cross_rate();
    /// set cross rate in variation              
    void set_cross_rate(float cross_rate);
    
    /// sets available functions based on comma-separated list.
    // void set_functions(const vector<string>& fns){ params.set_functions(fns); };
    // vector<string> get_functions(){return params.get_functions();};
                
    ///return max_depth of programs
    int get_max_depth();
    /// set max depth of programs              
    void set_max_depth(unsigned int max_depth);
      
    ///return max dimensionality of programs
    int get_max_size();
    /// set maximum sizeensionality of programs              
    void set_max_size(unsigned int max_dim);
    
    /// set seeds for each core's random number generator              
    // void set_random_state(int random_state);
    // int get_random_state() { return params.random_state; };
    // /// returns the actual seed determined by the input argument.
    // int get_random_state_() { return r.get_seed(); };
                
    ///return fraction of data to use for training
    float get_split();
    /// set train fraction of dataset
    void set_split(float sp);

    // int get_batch_size(){return params.bp.batch_size;};
    // void set_batch_size(int bs);
      
    ///set number of threads
    // void set_n_jobs(unsigned t);
    // int get_n_jobs(){return omp_get_num_threads();};
    
    ///set flag to use batch for training
    void set_use_batch();
    
    // getters and setters for the best solution found after evolution
    // predict, transform, predict_proba, etc.
    // get statistics
    // load and save best individuals
    // logger, save to file
    // execution archive
    // random state control
    // score functions
    // fit methods (this will run the evolution), run a single generation 

    bool is_fitted; ///< keeps track of whether fit was called.
private:
    Parameters params;  ///< hyperparameters of Feat 
    // attributes (hyperparameters)
    // update best
    // calculate/print stats
};

int main(){
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  auto [A, B, C, D] = taskflow.emplace(  // create four tasks
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; } 
  );                                  
                                      
  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C
                                      
  executor.run(taskflow).wait(); 

  return 0;
}

} // Brush

#endif
