#include "cbrush.h"
#include <iostream>


namespace Brush{

/// @brief initialize Feat object for fitting.
template <ProgramType T>
void CBrush<T>::init()
{
    if (params.n_jobs!=0) // TODO: change this to set taskflow jobs
        omp_set_num_threads(params.n_jobs);
    r.set_seed(params.random_state);

    // set up the pop, variator, etc
    set_is_fitted(false);

    // TODO: INIT SEARCH SPACE AND VARIATION HERE

    // TODO: implement stuff below
    // // start the clock
    // timer.Reset();

    // // signal handler
    // signal(SIGINT, my_handler);

    // // reset statistics
    // this->stats = Log_Stats();

    // params.use_batch = params.bp.batch_size>0;    

    // TODO: initialize dataset and search space here or inside fit?

}

template <ProgramType T>
bool CBrush<T>::update_best(const Dataset& data, bool val)
{
    float bs;
    bs = this->best_loss; 
    
    float f; 
    vector<Individual<T>>& pop_ref =this->pop.individuals; // TODO: archive here?

    bool updated = false; 

    for (const auto& ind: pop_ref)
    {
        if (ind.rank == 1)
        {
            if (val)
                f = ind.fitness_v;
            else
                f = ind.fitness;

            if (f < bs 
                || (f == bs && ind.get_complexity() < this->best_complexity)
                )
            {
                bs = f;
                this->best_ind = ind; 
                this->best_complexity = ind.get_complexity();

                updated = true;
            }
        }
    }

    this->best_loss = bs; 

    return updated;
}


template <ProgramType T>
void CBrush<T>::run_generation(unsigned int g, Dataset &data)
{
    // https://taskflow.github.io/taskflow/ParallelIterations.html
    tf::Executor executor;
    tf::Taskflow taskflow; // TODO: how to set number of threads?

    // TODO: implement custom behavior for first generation (specially regarding evaluator)
    params.current_gen = g;

    auto batch = data.get_batch(); // will return the original dataset if it is set to dont use batch 

    vector<vector<size_t>> island_parents;      
    island_parents.resize(pop.n_islands);
    taskflow.for_each_index(0, pop.n_islands,  1, [&](int island) {
        tuple<size_t, size_t> island_range = pop.get_island_range(island);

        // fit the weights with all training data
        evaluator.fitness(pop, island_range, data, params, true, false);
        evaluator.validation(pop, island_range, data, params, false);
    
        // TODO: if using batch, fitness should be called before selection to set the batch
        if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
            evaluator.fitness(pop, island_range, batch, params, false, false);

        // select parents
        vector<size_t> parents = selector.select(pop, island_range, params, data);
        island_parents.at(island) = parents;
    });
    
    vector<size_t> survivors(pop.size());
    pop.add_offspring_indexes();

    taskflow.for_each_index(0, pop.n_islands,  1, [&](int island) {
        tuple<size_t, size_t> island_range = pop.get_island_range(island);

        // // variation to produce offspring
        variator.vary(pop, island_range, island_parents.at(island));

        evaluator.fitness(pop, island_range, data, params, true, true);
        evaluator.validation(pop, island_range, data, params, true);

        if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
            evaluator.fitness(pop, island_range, batch, params, false, true);

        // select survivors from combined pool of parents and offspring
        auto island_survivors = survivor.survive(pop, island_range, params, data);
        
        auto [idx_start, idx_end] = island_range;
        size_t delta = idx_end - idx_start;
        for (unsigned i = 0; i<delta/2; ++i)
        {
            survivors.at(idx_start + (delta/2) + i) = island_survivors.at(i);
        }
    });

    // // reduce population to survivors
    pop.update(survivors);
    // pop.migrate();
    bool updated_best = update_best(data);
}

template <ProgramType T>
void CBrush<T>::fit(MatrixXf& X, VectorXf& y)
{
    this->init();

    // TODO: fit method that takes different arguments?
    Dataset data(X, y);

    this->ss = SearchSpace(data, params.functions);
    this->pop = Population(params.pop_size, params.num_islands);
    this->evaluator = Evaluation(params.scorer_);
    this->selector = Selection(params.sel, false);
    this->survivor = Selection(params.surv, true);

    // TODO: initialize (set operator) for survivor and selector
    // initialize population with initial model and/or starting pop
    pop.init(this->ss, this->params);

    unsigned g = 0;
    // continue until max gens is reached or max_time is up (if it is set)
    
    while(g<params.gens)      
    {
        run_generation(g, data);
    }

    set_is_fitted(true);
}

}