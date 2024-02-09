#ifndef SCORER_H
#define SCORER_H

#include "metrics.h"
#include "../util/error.h"
#include "../types.h"

// code to evaluate GP programs.
namespace Brush{
namespace Eval{


template <ProgramType P> // requires(P == PT::Regressor || P == PT::BinaryClassifier)
class Scorer
{

typedef float (*funcPointer)(const VectorXf&, 
                             const VectorXf&,
                             VectorXf&,
                             const vector<float>&);
public:
    // map the string into a function to be called when calculating the score
    std::map<string, funcPointer> score_hash;
    string scorer;

    // TODO: add more scores, include them here, add to score_hash
    Scorer(string scorer="mse") {
        // TODO: use this idea of map functpointer to do the mutations
        score_hash["mse"] = &mse;
        score_hash["log"] = &mean_log_loss; 
        // score_hash["multi_log"] = &mean_multi_log_loss; 
    
        this->set_scorer(scorer);
    };

    void set_scorer(string scorer){ this->scorer = scorer; };

    /* void set_scorer(string scorer); */
    float score(const VectorXf& y_true, const VectorXf& y_pred,
                VectorXf& loss, const vector<float>& w)
    {
        // loss is an array passed by reference to store each prediction (used in lexicase)
        // weights are used to give more or less importance for a given sample.
        // Every scorer must have the same function signature, but arent required to use all info
    
        if ( score_hash.find(this->scorer) == score_hash.end() ) 
        {
            // not found
            HANDLE_ERROR_THROW("Scoring function '" + this->scorer
                    + "' not defined");
            return 0.0;
        } 
        else 
        {
            // found
            return score_hash.at(this->scorer)(y_true, y_pred, loss, w); 
        }
    };
};



template <ProgramType P>
    requires( P == PT::MulticlassClassifier || P == PT::Representer)
class Scorer<P>
{

typedef float (*funcPointer)(const VectorXf&, 
                             const ArrayXXf&,
                             VectorXf&,
                             const vector<float>&);
public:
    // map the string into a function to be called when calculating the score
    std::map<string, funcPointer> score_hash;
    string scorer;

    Scorer(string scorer="multi_log") {
        score_hash["multi_log"] = &mean_multi_log_loss; 
    
        this->set_scorer(scorer);
    };

    void set_scorer(string scorer){ this->scorer = scorer; };

    /* void set_scorer(string scorer); */
    float score(const VectorXf& y_true, const ArrayXXf& y_pred,
                VectorXf& loss, const vector<float>& w)
    {
        // loss is an array passed by reference to store each prediction (used in lexicase)
        // weights are used to give more or less importance for a given sample.
        // Every scorer must have the same function signature, but arent required to use all info
    
        if ( score_hash.find(this->scorer) == score_hash.end() ) 
        {
            // not found
            HANDLE_ERROR_THROW("Scoring function '" + this->scorer
                    + "' not defined");
            return 0.0;
        } 
        else 
        {
            // found
            return score_hash.at(this->scorer)(y_true, y_pred, loss, w); 
        }
    };
};
}
}
#endif
