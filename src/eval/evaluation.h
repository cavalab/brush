
#ifndef EVALUATION_H
#define EVALUATION_H

#include <string>

#include "../search_space.h"
#include "../individual.h"
#include "../program/program.h"
#include "../data/data.h"

using std::string;

namespace Brush {

using namespace Pop;

namespace Eval {

template<ProgramType T> 
class Evaluation {
public:
    Evaluation(string scorer="");
    ~Evaluation();
        
    // TODO: IMPLEMENT THIS
    /// validation of population.
    void validation(vector<Individual<T>>& individuals,
                    const Dataset& data, 
                    const Parameters& params, 
                    bool offspring = false
                    );


    // TODO: EVALUATOR CALCULATE ERROR BASED ON TEMPLATING

    /// fitness of population.
    void fitness(vector<Individual<T>>& individuals,
                    const Dataset& data, 
                    const Parameters& params, 
                    bool offspring = false
                    );
    
    // TODO: implement other eval methods
    /// assign fitness to an individual.  
    // void assign_fit(Individual<T>& ind,
    //         const Dataset& data, 
    //         const Parameters& params,bool val=false);       

    // Scorer S;
};

} //selection
} //brush
#endif
