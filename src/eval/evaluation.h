
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
/**
 * @class Evaluation
 * @brief Class for evaluating the fitness of individuals in a population.
 */
class Evaluation {
public:
    Scorer<T> S;
    /**
     * @brief Constructor for Evaluation class.
     * @details Initializes the scorer based on the program type.
     */
    Evaluation(){
        // TODO: make eval update loss_v accordingly, and set to th same as train loss if there is no batch or no validation
    
        // TODO: make accuracy the main classification metric?
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
        
    /**
     * @brief Set the scorer for evaluation.
     * @param scorer The scorer to be set.
     */
    void set_scorer(string scorer){this->S.set_scorer(scorer);};

    /**
     * @brief Get the current scorer.
     * @return The current scorer.
     */
    string get_scorer(){return this->S.get_scorer();};
    
    /**
     * @brief Update the fitness of individuals in a population.
     * @param pop The population to update.
     * @param island The island index.
     * @param data The dataset for evaluation.
     * @param params The parameters for evaluation.
     * @param fit Flag indicating whether to update fitness.
     * @param validation Flag indicating whether to perform validation.
     */
    void update_fitness(Population<T>& pop,
                 int island,
                 const Dataset& data, 
                 const Parameters& params, 
                 bool fit=true,
                 bool validation=true // if there is no validation, then loss_v==loss
                 );
    
    /**
     * @brief Assign fitness to an individual.
     * @param ind The individual to assign fitness to.
     * @param data The dataset for evaluation.
     * @param params The parameters for evaluation.
     * @param val Flag indicating whether it is validation fitness.
     */
    void assign_fit(Individual<T>& ind, const Dataset& data,
                    const Parameters& params, bool val=false);

    // representation program (TODO: implement)
};

} //selection
} //brush
#endif
