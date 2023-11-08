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
void CBrush<T>::run_generation(unsigned int g,
                            vector<size_t> survivors,
                            Dataset &data,
                            float fraction,
                            unsigned& stall_count)
{
    params.current_gen = g;

    // select parents
    vector<size_t> parents = selector.select(pop, pop.get_island_range(0), data);
    
    // // variation to produce offspring
    variator.vary(pop, pop.get_island_range(0), parents);

    // TODO: needs to create the evaluator

    // // select survivors from combined pool of parents and offspring
    survivors = survivor.survive(pop, pop.get_island_range(0), data);
   
    // // reduce population to survivors
    pop.update(survivors);
    
    // bool updated_best = update_best(d);
}

}