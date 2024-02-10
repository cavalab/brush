
#ifndef EVALUATION_H
#define EVALUATION_H

#include <string>

#include "../search_space.h"
#include "../individual.h"
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
    Scorer<T> S;

    Evaluation(){
        string scorer;
        if ( (T == Brush::ProgramType::MulticlassClassifier)
        ||   (T == Brush::ProgramType::Representer) )
            scorer = "multi_log";
        else if (T == Brush::ProgramType::BinaryClassifier)
            scorer = "log";
        else 
            scorer = "mse";

        this->S.set_scorer(scorer);
    };
    ~Evaluation(){};
        
    void set_scorer(string scorer){this->S.set_scorer(scorer);};
    string get_scorer(){return this->S.get_scorer();};
    // TODO: set objectives
    // TODO: evaluation bind 
    // TODO: EVALUATOR CALCULATE ERROR BASED ON TEMPLATING? (caps)
    // TODO: MAKE it work for classification (do I need to have a way to set accuracy as a minimization problem?)
    /// fitness of population.
    void update_fitness(Population<T>& pop,
                 int island,
                 const Dataset& data, 
                 const Parameters& params, 
                 bool fit=true,
                 bool offspring = false,
                 bool validation=false
                 );
    
    /// assign fitness to an individual.
    void assign_fit(Individual<T>& ind, const Dataset& data,
                    const Parameters& params, bool val=false);

    // representation program (TODO: implement)
};

} //selection
} //brush
#endif
