#include "evaluation.h"

namespace Brush{   
namespace Eval{


// fitness of population
template<ProgramType T> 
void Evaluation<T>::update_fitness(Population<T>& pop,
                    int island,
                    const Dataset& data, 
                    const Parameters& params, 
                    bool fit,
                    bool validation
                    )
{   
    auto indices = pop.get_island_indexes(island);

    for (unsigned i = 0; i<indices.size(); ++i)
    {
        Individual<T>& ind = *pop.individuals.at(indices.at(i)).get(); // we are modifying it, so operator[] wont work

        bool pass = false;

        if (pass)
        {
            ind.fitness.set_loss(MAX_FLT);
            ind.fitness.set_loss_v(MAX_FLT);

            ind.error = MAX_FLT*VectorXf::Ones(data.y.size());
        }
        else
        {
            // assign weights to individual
            if (fit && ind.get_is_fitted() == false)
            {
                ind.program.fit(data);
            }
            
            assign_fit(ind, data, params, validation);
        }
    }
}

// assign loss to program
template<ProgramType T> 
void Evaluation<T>::assign_fit(Individual<T>& ind, const Dataset& data, 
                               const Parameters& params, bool val)
{
    VectorXf errors;
    using PT = ProgramType;
    
    Dataset train = data.get_training_data();
    float f = S.score(ind, train, errors, params);
    
    float f_v = f;
    if (data.use_validation) {
        Dataset validation = data.get_validation_data();
        f_v = S.score(ind, validation, errors, params);
    }

    // TODO: implement the class weights and use it here (and on errors)

    // This is what is going to determine the weights for the individual's fitness
    ind.set_objectives(params.get_objectives());

    // we will always set all values for fitness (regardless of being used).
    // this will make sure the information is calculated and ready to be used
    // regardless of how the program is set to run.
    ind.error = errors;

    // when we use these setters, it updates its previous values references
    ind.fitness.set_loss(f);
    ind.fitness.set_loss_v(f_v);
    ind.fitness.set_size(ind.get_size());
    ind.fitness.set_complexity(ind.get_complexity());
    ind.fitness.set_depth(ind.get_depth());

    vector<float> values;
    values.resize(0);

    for (const auto& n : ind.get_objectives())
    {
        if (n.compare(params.scorer)==0)
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