
#ifndef EVALUATION_H
#define EVALUATION_H

#include <string>

#include "../vary/search_space.h"
#include "../ind/individual.h"
#include "../data/data.h"
#include "scorer.h"
#include "../pop/population.h"

using std::string;

namespace Brush {

using namespace Pop;

namespace Eval {

template<ProgramType T> 
class Evaluation {
public:
    Scorer<T> S;

    // TODO: make eval update loss_v accordingly, and set to th same as train loss if there is no batch or no validation
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
    
    /// fitness of population.
    void update_fitness(Population<T>& pop,
                 int island,
                 const Dataset& data, 
                 const Parameters& params, 
                 bool fit=true,
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
