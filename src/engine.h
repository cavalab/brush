/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef Engine_H
#define Engine_H

#include "util/rnd.h"
#include "init.h"
#include "params.h"
#include "eval/evaluation.h"
#include "vary/variation.h"
#include "pop/population.h"
#include "pop/archive.h"
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
/**
 * @brief The `Engine` class represents the core engine of the brush library.
 * 
 * It encapsulates the functionality for training and predicting with programs
 * in a genetic programming framework. The `Engine` class manages the population
 * of programs, selection algorithms, evaluation code, variation operators, and
 * survival algorithms. It also provides methods for training the model, making
 * predictions, and accessing runtime statistics.
 * 
 * The `Engine` class is parameterized by the program type `T`, which determines
 * the type of programs that can be evolved and evaluated by the engine.
 */
class Engine{
public:
    Engine()
    {
        this->params = Parameters();
        this->ss = SearchSpace();
    };

    Engine(Parameters& p, SearchSpace& s)
    {
        this->params = p;
        this->ss = s;
        // TODO: make variation to have a default constructor
        // this->variator(Variation<T>(params, ss)) ;
    };
    
    ~Engine(){};

    // outputs a progress bar, filled according to @param percentage.
    void print_progress(float percentage);
    void calculate_stats();
    void print_stats(std::ofstream& log, float fraction);      
    void log_stats(std::ofstream& log);

    // all hyperparameters are controlled by the parameter class. please refer to that to change something
    inline Parameters& get_params(){return params;}
    inline void set_params(Parameters& p){params=p;}

    inline SearchSpace& get_search_space() { return ss; }
    inline void set_search_space(SearchSpace& space) { ss = space; }

    inline bool get_is_fitted(){return is_fitted;}

    /// updates best score by searching in the population for the individual that best fits the given data
    bool update_best();

    Individual<T>& get_best_ind(){return best_ind;};  
    
    Engine<T> &fit(Dataset& data) {
        run(data);
        return *this;
    };
    Engine<T> &fit(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)
    {
        // Using constructor 2 to create the dataset
        Dataset d(X,y,params.feature_names,{},params.feature_types,
                params.classification,params.validation_size,
                params.batch_size, params.shuffle_split);
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

    // TODO: predict/predict_proba/archive with longitudinal data

    /// train the model
    void run(Dataset &d);
    
    // TODO: should params and ss be private? (that would require better json handling)
    Parameters params;  ///< hyperparameters of brush, which the user can interact
    SearchSpace ss;
    
    Individual<T> best_ind; ///< best individual found during training
    Archive<T> archive;     ///< pareto front archive

    bool is_fitted; ///< keeps track of whether fit was called
private:

    Population<T> pop;       	///< population of programs
    Selection<T>  selector;   ///< selection algorithm
    Evaluation<T> evaluator;  ///< evaluation code
    Variation<T>  variator;  	///< variation operators
    Selection<T>  survivor;   ///< survival algorithm
    
    Log_Stats stats; ///< runtime stats

    Timer timer; ///< start time of training

    void init();

    /// set flag indicating whether fit has been called
    inline void set_is_fitted(bool f){is_fitted=f;}
};

// TODO: should I serialize data and search space as well?
// Only stuff to make new predictions should be serialized
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Engine<PT::Regressor>, params, best_ind, archive, ss, is_fitted);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Engine<PT::BinaryClassifier>, params, best_ind, archive, ss, is_fitted);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Engine<PT::MulticlassClassifier>, params, best_ind, archive, ss, is_fitted);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Engine<PT::Representer>, params, best_ind, archive, ss, is_fitted);

} // Brush
#endif
