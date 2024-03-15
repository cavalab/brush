/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef Engine_H
#define Engine_H

#include "./util/rnd.h"
#include "init.h"
#include "params.h"
#include "population.h"
#include "./eval/evaluation.h"
#include "variation.h"
#include "selection/selection.h"
#include "taskflow/taskflow.hpp"

#include <taskflow/algorithm/for_each.hpp>

namespace Brush
{

using namespace Pop;
using namespace Sel;
using namespace Eval;
using namespace Var;

template <ProgramType T>
class Engine{
public:
    Engine(const Parameters& p=Parameters())
    : params(p)
    , ss(SearchSpace()) // we need to initialize ss and variator. TODO: make them have a default way so we dont have to initialize here
    , variator(Variation<T>(params, ss)) 
    {};
    
    ~Engine(){};

    // all hyperparameters are controlled by the parameter class. please refer to that to change something
    inline Parameters& get_params(){return params;}
    inline void set_params(Parameters& p){params=p;}

    inline bool get_is_fitted(){return is_fitted;}

    /// updates best score by searching in the population for the individual that best fits the given data
    bool update_best(const Dataset& data, bool val=false);  

    // TODO: best fitness instead of these. use fitness comparison
    float best_loss;
    int best_complexity;
    Individual<T>& get_best_ind(){return best_ind;};  
    
    /// train the model
    void run(Dataset &d);
    
    Parameters params;  ///< hyperparameters of brush, which the user can interact
private:
    SearchSpace ss;

    Population<T> pop;       	///< population of programs
    Selection<T>  selector;   ///< selection algorithm
    Evaluation<T> evaluator;  ///< evaluation code
    Variation<T>  variator;  	///< variation operators
    Selection<T>  survivor;   ///< survival algorithm
    
    // TODO: MISSING CLASSES: timer, archive, logger
    Individual<T> best_ind;
    bool is_fitted; ///< keeps track of whether fit was called.

    void init();

    /// set flag indicating whether fit has been called
    inline void set_is_fitted(bool f){is_fitted=f;}

    // TODO: calculate/print stats
};

} // Brush

#endif
