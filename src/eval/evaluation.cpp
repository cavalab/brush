#include "evaluation.h"

namespace Brush{   
namespace Eval{


template<ProgramType T> 
void Evaluation<T>::validation(Population<T>& pop,
                    int island, 
                    const Dataset& data, 
                    const Parameters& params, 
                    bool offspring
                    )
{
    auto idxs = pop.get_island_indexes(island);

    int start = 0;
    if (offspring)
        start = idxs.size()/2;

    for (unsigned i = start; i<idxs.size(); ++i)
    {
        Individual<T>& ind = *pop.individuals.at(idxs.at(i)).get(); // we are modifying it, so operator[] wont work
        
        // if there is no validation data,
        // set fitness_v to fitness and return ( this assumes that fitness on train was calculated previously.)
        if (!data.use_validation) 
        {
            ind.fitness_v = ind.fitness;
            continue;
        }

        bool pass = true;

        if (!pass)
        {
            // TODO: stop doing this hardcoded?
            ind.fitness_v = MAX_FLT; 
        }
        else
        {
            // TODO: implement the class weights and use it here (and on fitness)
            VectorXf y_pred =  ind.program.predict(data.get_validation_data());
            assign_fit(ind, y_pred, data, params, true);
        }
    }
}

// fitness of population
template<ProgramType T> 
void Evaluation<T>::fitness(Population<T>& pop,
                    int island,
                    const Dataset& data, 
                    const Parameters& params, 
                    bool fit,
                    bool offspring
                    )
{
    auto idxs = pop.get_island_indexes(island);

    int start = 0;
    if (offspring)
        start = idxs.size()/2;

    for (unsigned i = start; i<idxs.size(); ++i)
    {
        Individual<T>& ind = *pop.individuals.at(idxs.at(i)).get(); // we are modifying it, so operator[] wont work

        bool pass = true;

        if (!pass)
        {
            ind.fitness = MAX_FLT;
            ind.error = MAX_FLT*VectorXf::Ones(data.y.size());
        }
        else
        {
            // assign weights to individual
            if (fit)
                ind.program.fit(data);
            
            VectorXf y_pred =  ind.program.predict(data.get_training_data());
            assign_fit(ind, y_pred, data, params, false);
        }
    }
}

// assign fitness to program
template<ProgramType T> 
void Evaluation<T>::assign_fit(Individual<T>& ind,  
        VectorXf& y_pred, const Dataset& data, 
        const Parameters& params, bool val)
{
    VectorXf loss;

    float f = S.score(data.y, y_pred, loss, params.class_weights);
    
    if (val)
    {
        ind.fitness_v = f;
    }
    else
    {
        ind.fitness = f;
        ind.error = loss;
    }
}

} // Pop
} // Brush