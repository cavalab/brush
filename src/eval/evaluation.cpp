#include "evaluation.h"

namespace Brush{   
namespace Eval{


template<ProgramType T> 
void Evaluation<T>::validation(Population<T>& pop,
                    tuple<size_t, size_t> island_range, 
                    const Dataset& data, 
                    const Parameters& params, 
                    bool offspring
                    )
{
    // if offspring false --> if has offspring, do it on first half. else, do on entire island
    // offspring true --> assert that has offspring, do it on the second half of the island

    auto [idx_start, idx_end] = island_range;
    size_t delta = idx_end - idx_start;
    if (offspring)
    {
        assert(pop.offspring_ready
            && ("Population does not have offspring to calculate validation fitness"));
        
        idx_start = idx_start + (delta/2);
    }
    else if (pop.offspring_ready) // offspring is false. We need to see where we sould stop
    {
        idx_end = idx_end - (delta/2);
    }

    for (unsigned i = idx_start; i<idx_end; ++i)
    {
        Individual<T>& ind = pop[i];
        
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
                    tuple<size_t, size_t> island_range, 
                    const Dataset& data, 
                    const Parameters& params, 
                    bool fit,
                    bool offspring
                    )
{
    // if offspring false --> if has offspring, do it on first half. else, do on entire island
    // offspring true --> assert that has offspring, do it on the second half of the island

    auto [idx_start, idx_end] = island_range;
    size_t delta = idx_end - idx_start;
    if (offspring)
    {
        assert(pop.offspring_ready
            && ("Population does not have offspring to calculate validation fitness"));
        
        idx_start = idx_start + (delta/2);
    }
    else if (pop.offspring_ready) // offspring is false. We need to see where we sould stop
    {
        idx_end = idx_end - (delta/2);
    }

    for (unsigned i = idx_start; i<idx_end; ++i)
    {
        Individual<T>& ind = pop.individuals.at(i);

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