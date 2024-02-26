#include "evaluation.h"

namespace Brush{   
namespace Eval{


// TODO: merge validation and update fitness into one function
// fitness of population
template<ProgramType T> 
void Evaluation<T>::update_fitness(Population<T>& pop,
                    int island,
                    const Dataset& data, 
                    const Parameters& params, 
                    bool fit,
                    bool offspring,
                    bool validation
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
            // TODO: better handling of nan or inf scores when doing selection and survival (and hall of fame and rank for migration)
            ind.fitness.loss = MAX_FLT;
            ind.fitness.loss_v = MAX_FLT;
            ind.error = MAX_FLT*VectorXf::Ones(data.y.size());
        }
        else
        {
            // assign weights to individual
            if (fit)
                ind.program.fit(data);
            
            assign_fit(ind, data, params, validation);
        }
    }
}

// assign loss to program
template<ProgramType T> 
void Evaluation<T>::assign_fit(Individual<T>& ind, const Dataset& data, 
                               const Parameters& params, bool val)
{
    VectorXf loss;
    using PT = ProgramType;
    
    Dataset validation = data.get_validation_data();
    float f_v = S.score(ind, validation, loss, params);

    // TODO: implement the class weights and use it here (and on loss)

    Dataset train = data.get_training_data();
    float f = S.score(ind, train, loss, params);
    
    // TODO: setter for loss and loss_v
    ind.error = loss;
    ind.fitness.loss = f;
    ind.fitness.loss_v = f_v;
    ind.fitness.size = ind.program.size();
    ind.fitness.complexity = ind.program.complexity();
    ind.fitness.depth = ind.program.depth();

    ind.set_objectives(params.objectives);

    vector<float> values;
    values.resize(0);

    for (const auto& n : ind.get_objectives())
    {
        if (n.compare("error")==0)
            values.push_back(val ? f_v : f);
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