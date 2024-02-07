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
        // set loss_v to loss and return ( this assumes that loss on train was calculated previously.)
        if (!data.use_validation) 
        {
            ind.loss_v = ind.loss;
            continue;
        }

        bool pass = true;

        if (!pass)
        {
            // TODO: stop doing this hardcoded?
            ind.loss_v = MAX_FLT; 
        }
        else
        {
            // TODO: implement the class weights and use it here (and on loss)
            VectorXf y_pred =  ind.program.predict(data.get_validation_data());
            assign_fit(ind, y_pred, data, params, true);
        }
        // ind.set_obj(params.objectives);
    }
}

// fitness of population
template<ProgramType T> 
void Evaluation<T>::update_fitness(Population<T>& pop,
                    int island,
                    const Dataset& data, 
                    const Parameters& params, 
                    bool fit,
                    bool offspring
                    )
{
    //TODO:  it could use the validation_loss     
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
            // TODO: check if score was nan and assign the max float
            ind.fitness.loss = MAX_FLT;
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

// assign loss to program
template<ProgramType T> 
void Evaluation<T>::assign_fit(Individual<T>& ind,  
        VectorXf& y_pred, const Dataset& data, 
        const Parameters& params, bool val)
{
    VectorXf loss;

    float f = S.score(data.y, y_pred, loss, params.class_weights);
    
    if (val)
    {   // TODO: use this function to decide wether to take loss from validation or training
        ind.fitness.loss_v = f;
    }
    else
    {
        // TODO: setter for loss  and loss_v
        ind.fitness.loss = f;
        ind.error = loss;
    }
    ind.fitness.size = ind.program.size();
    ind.fitness.complexity = ind.program.complexity();
    ind.fitness.depth = ind.program.depth();

    ind.set_objectives(params.objectives);

    vector<float> values;
    values.resize(0);

    for (const auto& n : ind.get_objectives())
    {
        if (n.compare("error")==0)
            values.push_back(f); // fitness on training data, not validation.
                                    // if you use batch, this value will change every generation
        else if (n.compare("complexity")==0)
            values.push_back(ind.program.complexity());
        else if (n.compare("size")==0)
            values.push_back(ind.program.size());
        else if (n.compare("depth")==0)
            values.push_back(ind.program.depth());
        else
            HANDLE_ERROR_THROW(n+" is not a known objective");
    }
    
    // will use inner attributes to set the fitness object
    ind.fitness.set_values(values); 
}

} // Pop
} // Brush