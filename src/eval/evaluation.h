
#ifndef EVALUATION_H
#define EVALUATION_H

#include <string>

#include "../search_space.h"
#include "../individual.h"
#include "../program/program.h"
#include "../data/data.h"
#include "scorer.h"
#include "../population.h"

using std::string;

namespace Brush {

using namespace Pop;

namespace Eval {

template<ProgramType T> 
class Evaluation {
public:
    Scorer S;

    Evaluation(string scorer="mse"): S(scorer) { this->S.set_scorer(scorer); };
    ~Evaluation(){};
        
    /// validation of population.
    void validation(Population<T>& pop,
                    int island, 
                    const Dataset& data, 
                    const Parameters& params, 
                    bool offspring = false
                    );

    // TODO: EVALUATOR CALCULATE ERROR BASED ON TEMPLATING? (caps)
    // TODO: MAKE it work for classification (do I need to have a way to set accuracy as a minimization problem?)
    /// fitness of population.
    void fitness(Population<T>& pop,
                 int island,
                 const Dataset& data, 
                 const Parameters& params, 
                 bool fit=true,
                 bool offspring = false
                 );
    
    // TODO: implement other eval methods

    /// assign fitness to an individual.  
    void assign_fit(Individual<T>& ind, VectorXf& y_pred,
            const Dataset& data, const Parameters& params, bool val=false);       

};

} //selection
} //brush
#endif
