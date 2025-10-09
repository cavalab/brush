#ifndef SCORER_H
#define SCORER_H

#include "metrics.h"
#include "../util/error.h"
#include "../types.h"

// code to evaluate GP programs.
namespace Brush{

using namespace Pop;

namespace Eval{


template <ProgramType P>
class Scorer
{

using RetType =
        typename std::conditional_t<P == PT::Regressor, ArrayXf,
                    std::conditional_t<P == PT::Representer, ArrayXXf, ArrayXf
        >>;
        
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
        score_hash["mse"] = &mse; 
    
        this->set_scorer(scorer);
    };

    void set_scorer(string scorer){ this->scorer = scorer; };
    string get_scorer(){return this->scorer; };

    /* void set_scorer(string scorer); */
    float score(const VectorXf& y_true, const VectorXf& y_pred,
                VectorXf& loss, const vector<float>& w)
    {
        // loss is an array passed by reference to store each prediction (used in lexicase)
        // weights are used to give more or less importance for a given sample.
        // Every scorer must have the same function signature, but arent required to use all info
    
        if ( score_hash.find(this->scorer) == score_hash.end() ) 
        {
            HANDLE_ERROR_THROW("Scoring function '" + this->scorer + "' not defined");
            return 0.0;
        } 
        else 
        {
            return score_hash.at(this->scorer)(y_true, y_pred, loss, w); 
        }
    };

    float score(Individual<P>& ind, Dataset& data, 
                VectorXf& loss, const Parameters& params)
    {
        RetType y_pred = ind.predict(data);
        return score(data.y, y_pred, loss, params.class_weights);
    }
};


// TODO: improve this so we dont have a lot of different declarations
template <ProgramType P>
    requires( P == PT::BinaryClassifier)
class Scorer<P>
{

using RetType = ArrayXf;

typedef float (*funcPointer)(const VectorXf&, 
                             const VectorXf&,
                             VectorXf&,
                             const vector<float>&);
public:
    // map the string into a function to be called when calculating the score
    std::map<string, funcPointer> score_hash;
    string scorer;

    Scorer(string scorer="log") {
        score_hash["log"] = &mean_log_loss;
        score_hash["average_precision_score"] = &average_precision_score;
        score_hash["accuracy"] = &zero_one_loss;
        score_hash["balanced_accuracy"] = &bal_zero_one_loss;
    
        this->set_scorer(scorer);
    };

    void set_scorer(string scorer){ this->scorer = scorer; };
    string get_scorer(){return this->scorer; };

    /* void set_scorer(string scorer); */
    float score(const VectorXf& y_true, const VectorXf& y_pred,
                VectorXf& loss, const vector<float>& w)
    {
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

    float score(Individual<P>& ind, Dataset& data, 
                VectorXf& loss, const Parameters& params)
    {
        RetType y_pred = ind.predict_proba(data); // .template cast<float>();
        
        vector<float> class_weights = {};

        // calculate class weights based on current data --- instead of using a pre-calculated value.
        // This is only true for the scoring. For other usages of class weights, then
        // the training data is used.
        if (params.class_weights_type == "support")
        {
            class_weights.resize(params.n_classes);
            for (unsigned i = 0; i < params.n_classes; ++i){
                // weighting by support 
                int support = (data.y.cast<int>().array() == i).count();

                if (support==0)
                    class_weights.at(i) = 0.0f;
                else
                    class_weights.at(i) = float(data.y.size()) / float(params.n_classes * support);
            }
        }
        else // else it is either unbalanced or user_defined
        { 
            class_weights = params.class_weights;
        }
        
        return score(data.y, y_pred, loss, class_weights);
    }
};

template <ProgramType P>
    requires(P == PT::MulticlassClassifier)
class Scorer<P>
{

using RetType = ArrayXXf;

typedef float (*funcPointer)(const VectorXf&, 
                             const ArrayXXf&,
                             VectorXf&,
                             const vector<float>&);
public:
    // map the string into a function to be called when calculating the score
    std::map<string, funcPointer> score_hash;
    string scorer;

    // TODO: I need to test this stuff
    Scorer(string scorer="multi_log") {
        score_hash["multi_log"] = &mean_multi_log_loss; 
        score_hash["accuracy"] = &multi_zero_one_loss;
    
        this->set_scorer(scorer);
    };

    void set_scorer(string scorer){ this->scorer = scorer; };
    string get_scorer(){return this->scorer; };

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

    float score(Individual<P>& ind, Dataset& data, 
                VectorXf& loss, const Parameters& params)
    {
        RetType y_pred = ind.predict_proba(data); // .template cast<float>();
        
        vector<float> class_weights = {};
        if (params.class_weights_type == "support")
        {
            class_weights.resize(params.n_classes);
            for (unsigned i = 0; i < params.n_classes; ++i){
                // weighting by support 
                int support = (data.y.cast<int>().array() == i).count();

                if (support==0)
                    class_weights.at(i) = 0.0;
                else
                    class_weights.at(i) = float(data.y.size()) / float(params.n_classes * support);
            }
        } // or else it is either unbalanced or user_defined
        else {
            class_weights = params.class_weights;
        }

        return score(data.y, y_pred, loss, class_weights);
    }
};

}
}
#endif
