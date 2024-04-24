/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef Engine_H
#define Engine_H

#include "./util/rnd.h"
#include "init.h"
#include "params.h"
#include "pop/population.h"
#include "pop/archive.h"
#include "./eval/evaluation.h"
#include "vary/variation.h"
#include "selection/selection.h"
#include "taskflow/taskflow.hpp"

#include <taskflow/algorithm/for_each.hpp>

namespace Brush
{

using namespace Pop;
using namespace Sel;
using namespace Eval;
using namespace Var;
using namespace nlohmann;

template <ProgramType T>
class Engine{
public:
    Engine(const Parameters& p=Parameters())
    : params(p)
    , ss(SearchSpace()) // we need to initialize ss and variator. TODO: make them have a default way so we dont have to initialize here
    , variator(Variation<T>(params, ss)) 
    {};
    
    ~Engine(){};

    // outputs a progress bar, filled according to @param percentage.
    void print_progress(float percentage);
    void calculate_stats(const Dataset& d);
    void print_stats(std::ofstream& log, float fraction);      
    void log_stats(std::ofstream& log);

    // all hyperparameters are controlled by the parameter class. please refer to that to change something
    inline Parameters& get_params(){return params;}
    inline void set_params(Parameters& p){params=p;}

    inline bool get_is_fitted(){return is_fitted;}

    /// updates best score by searching in the population for the individual that best fits the given data
    bool update_best(const Dataset& data, bool val=false);  
    // TODO: hyperparameter to set how the best is picked (MCDM, best on val, pareto front, etc). one of the options should be getting the pareto front

    // TODO: best fitness instead of these. use fitness comparison
    float best_score;
    int best_complexity;
    Individual<T>& get_best_ind(){return best_ind;};  
    
    // TODO: starting pop (just like feat)

    // TODO: make thesqe work
    // /// predict on unseen data.             
    // VectorXf predict(MatrixXf& X, LongData& Z);  
    // VectorXf predict(MatrixXf& X);

    // /// predict on unseen data. return CLabels.
    // shared_ptr<CLabels> predict_labels(MatrixXf& X, LongData Z = LongData());  

    // /// predict probabilities of each class.
    // ArrayXXf predict_proba(MatrixXf& X, LongData& Z);  
    // ArrayXXf predict_proba(MatrixXf& X);

    // archive stuff
    // TODO: make these work
    ///return archive size
    int get_archive_size(){ return this->archive.individuals.size(); };
    ///return population as string
    vector<json> get_archive(bool front);
    
    // /// predict on unseen data from the whole archive             
    // VectorXf predict_archive(int id, MatrixXf& X);  
    // VectorXf predict_archive(int id, MatrixXf& X, LongData& Z);
    // ArrayXXf predict_proba_archive(int id, MatrixXf& X, LongData& Z);
    // ArrayXXf predict_proba_archive(int id, MatrixXf& X);


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
    
    Log_Stats stats; ///< runtime stats

    Timer timer;       ///< start time of training
    Archive<T> archive;          ///< pareto front archive

    Individual<T> best_ind;
    bool is_fitted; ///< keeps track of whether fit was called.

    void init();

    /// set flag indicating whether fit has been called
    inline void set_is_fitted(bool f){is_fitted=f;}
};

// TODO: serialization for engine with NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE

} // Brush

#endif
