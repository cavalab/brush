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
    void calculate_stats();
    void print_stats(std::ofstream& log, float fraction);      
    void log_stats(std::ofstream& log);

    // all hyperparameters are controlled by the parameter class. please refer to that to change something
    inline Parameters& get_params(){return params;}
    inline void set_params(Parameters& p){params=p;}

    inline bool get_is_fitted(){return is_fitted;}

    /// updates best score by searching in the population for the individual that best fits the given data
    bool update_best(const Dataset& data, bool val=false);  
    // TODO: hyperparameter to set how the best is picked (MCDM, best on val, pareto front, etc). one of the options should be getting the pareto front

    // TODO: best fitness (the class) instead of these. use fitness comparison
    float best_score;
    int best_complexity;
    Individual<T>& get_best_ind(){return best_ind;};  
    
    Engine<T> &fit(Dataset& data) {
        run(data);
        return *this;
    };
    Engine<T> &fit(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)
    {
        // Using constructor 2 to create the dataset
        Dataset d(X,y,params.feature_names,{},params.classification,
                params.validation_size, params.batch_size);
        return fit(d);
    };

    auto predict(const Dataset& data) { return this->best_ind.predict(data); };
    auto predict(const Ref<const ArrayXXf>& X)
    {
        Dataset d(X);
        return predict(d);
    };

    template <ProgramType P = T>
        requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
    auto predict_proba(const Dataset &d) { return this->best_ind.predict_proba(d); };
    template <ProgramType P = T>
        requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
    auto predict_proba(const Ref<const ArrayXXf>& X) 
    {
        Dataset d(X);
        return predict_proba(d);
    };

    // TODO: starting pop (just like feat)

    // TODO: make these work
    // /// predict on unseen data.             
    // VectorXf predict(MatrixXf& X, LongData& Z);  
    // VectorXf predict(MatrixXf& X);

    // /// predict on unseen data. return CLabels.
    // shared_ptr<CLabels> predict_labels(MatrixXf& X, LongData Z = LongData());  

    // /// predict probabilities of each class.
    // ArrayXXf predict_proba(MatrixXf& X, LongData& Z);  
    // ArrayXXf predict_proba(MatrixXf& X);

    // archive stuff ---

    ///return archive size
    int get_archive_size(){ return this->archive.individuals.size(); };

    ///return population as string
    vector<json> get_archive(bool front);
    
    /// predict on unseen data from the archive             
    auto predict_archive(int id, const Dataset& data);
    auto predict_archive(int id, const Ref<const ArrayXXf>& X);

    template <ProgramType P = T>
        requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
    auto predict_proba_archive(int id, const Dataset& data);
    template <ProgramType P = T>
        requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
    auto predict_proba_archive(int id, const Ref<const ArrayXXf>& X);

    // TODO: make these work
    // VectorXf predict_archive(int id, const Ref<const ArrayXXf>& X, LongData& Z);
    // ArrayXXf predict_proba_archive(int id, const Ref<const ArrayXXf>& X, LongData& Z);

    /// train the model
    void run(Dataset &d);
    
    Parameters params;  ///< hyperparameters of brush, which the user can interact
    Individual<T> best_ind;
    
    Archive<T> archive;          ///< pareto front archive
private:
    SearchSpace ss;

    Population<T> pop;       	///< population of programs
    Selection<T>  selector;   ///< selection algorithm
    Evaluation<T> evaluator;  ///< evaluation code
    Variation<T>  variator;  	///< variation operators
    Selection<T>  survivor;   ///< survival algorithm
    
    Log_Stats stats; ///< runtime stats

    Timer timer;       ///< start time of training

    bool is_fitted; ///< keeps track of whether fit was called.

    void init();

    /// set flag indicating whether fit has been called
    inline void set_is_fitted(bool f){is_fitted=f;}
};

// Only stuff to make new predictions or call fit again
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Engine<PT::Regressor>, params, best_ind, archive);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Engine<PT::BinaryClassifier>,params, best_ind, archive);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Engine<PT::MulticlassClassifier>,params, best_ind, archive);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Engine<PT::Representer>,params, best_ind, archive);

} // Brush
#endif
