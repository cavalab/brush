#include "cbrush.h"
#include <iostream>


namespace Brush{

/// @brief initialize Feat object for fitting.
void CBrush::init()
{
    if (params.n_jobs!=0) // TODO: change this to set taskflow jobs
        omp_set_num_threads(params.n_jobs);
    r.set_seed(params.random_state);

    set_is_fitted(false);

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

void CBrush::run_generation(unsigned int g,
                            vector<size_t> survivors,
                            Dataset &d,
                            float fraction,
                            unsigned& stall_count)
{
    // d.t->set_protected_groups();

    // params.set_current_gen(g);

    // // select parents
    // logger.log("selection..", 2);
    // vector<size_t> parents = selector.select(pop, params, *d.t);
    // logger.log("parents:\n"+pop.print_eqns(), 3);          
    
    // // variation to produce offspring
    // logger.log("variation...", 2);
    // variator.vary(pop, parents, params,*d.t);
    // logger.log("offspring:\n" + pop.print_eqns(true), 3);

    // // evaluate offspring
    // logger.log("evaluating offspring...", 2);
    // evaluator.fitness(pop.individuals, *d.t, params, true);
    // evaluator.validation(pop.individuals, *d.v, params, true);

    // // select survivors from combined pool of parents and offspring
    // logger.log("survival...", 2);
    // survivors = survivor.survive(pop, params, *d.t);
   
    // // reduce population to survivors
    // logger.log("shrinking pop to survivors...",2);
    // pop.update(survivors);
    // logger.log("survivors:\n" + pop.print_eqns(), 3);
    
    // logger.log("update best...",2);
    // bool updated_best = update_best(d);

    // logger.log("calculate stats...",2);
    // calculate_stats(d);

    // if (params.max_stall > 0)
    //     update_stall_count(stall_count, updated_best);

    // logger.log("update archive...",2);
    // if (use_arch) 
    //     archive.update(pop,params);
    
    // if(params.verbosity>1)
    //     print_stats(log, fraction);    
    // else if(params.verbosity == 1)
    //     printProgress(fraction);
    
    // if (!logfile.empty())
    //     log_stats(log);

    // if (save_pop > 1)
    //     pop.save(this->logfile+".pop.gen" + 
    //                 to_string(params.current_gen) + ".json");

    // // tighten learning rate for grad descent as evolution progresses
    // if (params.backprop)
    // {
    //     params.bp.learning_rate = \
    //         (1-1/(1+float(params.gens)))*params.bp.learning_rate;
    //     logger.log("learning rate: " 
    //             + std::to_string(params.bp.learning_rate),3);
    // }
    // logger.log("finished with generation...",2);
}

}