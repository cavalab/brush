#ifndef SCORER_H
#define SCORER_H

#include "metrics.h"
#include "../util/error.h"

// code to evaluate GP programs.
namespace Brush{
namespace Eval{

typedef float (*funcPointer)(const VectorXf&, 
                             const VectorXf&,
                             VectorXf&,
                             const vector<float>&);

class Scorer
{
public:
    // map the string into a function to be called when calculating the score
    std::map<string, funcPointer> score_hash;
    string scorer;

    // TODO: add more scores, include them here, add to score_hash
    Scorer(string scorer="mse") {
        score_hash["mse"] = &mse;
    
        this->set_scorer(scorer);
    };

    void set_scorer(string scorer){ this->scorer = scorer; };

    /* void set_scorer(string scorer); */
    float score(const VectorXf& y_true, VectorXf& y_pred,
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

    // overloaded score with no loss
    float score(const VectorXf& y_true, VectorXf& y_pred,
                vector<float> w=vector<float>())
    {
        VectorXf dummy;
        return this->score(y_true, y_pred, dummy, w);
    };
};

}
}
#endif
