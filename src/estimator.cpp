#include "estimator.h"


#include <iostream>


namespace Brush{


using namespace Pop;
using namespace Sel;
using namespace Eval;
using namespace Var;

/// @brief initialize Feat object for fitting.
template <ProgramType T>
void Estimator<T>::init()
{
    std::cout << "inside init" << std::endl;

    // TODO: initialize (set operator) for survivor and selector
    // initialize population with initial model and/or starting pop

    if (params.n_jobs!=0) // TODO: change this to set taskflow jobs
        omp_set_num_threads(params.n_jobs);

    std::cout << "set number of threads" << std::endl;

    r.set_seed(params.random_state);
    std::cout << "set random state" << std::endl;


    // set up the pop, variator, etc
    set_is_fitted(false);
    std::cout << "is fitted is false" << std::endl;


    this->pop = Population<T>();
    std::cout << "created population" << std::endl;

    this->evaluator = Evaluation<T>();
    std::cout << "created evaluator" << std::endl;

    this->selector = Selection<T>(params.sel, false);
    std::cout << "created selector" << std::endl;

    this->survivor = Selection<T>(params.surv, true);
    std::cout << "created survivor" << std::endl;

    //TODO
    ///return fraction of data to use for training
    // float get_split();
    // /// set train fraction of dataset
    // void set_split(float sp);

    // TODO
    // int get_batch_size(){return params.bp.batch_size;};
    // void set_batch_size(int bs);
      
    // TODO
    ///set number of threads (and use them in taskflow)
    // void set_n_jobs(unsigned t);
    // int get_n_jobs(){return omp_get_num_threads();};
    
    ///set flag to use batch for training
    // void set_use_batch();
    
    // TODO getters and setters for the best solution found after evolution
    // predict, transform, predict_proba, etc.
    // get statistics
    // load and save best individuals
    // logger, save to file
    // execution archive
    // score functions
    // fit methods (this will run the evolution)

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

template <ProgramType T> // TODO: use the dataset, or ignore it
bool Estimator<T>::update_best(const Dataset& data, bool val)
{
    std::cout << "updating best" << std::endl;

    float bs;
    bs = this->best_loss; 
    
    float f; 
    // TODO: archive here?

    bool updated = false; 

    std::cout << "inside loop" << std::endl;

    vector<size_t> hof = this->pop.hall_of_fame(1);

    std::cout << "got hof" << std::endl;

    // will look only in the first half of the population (this is intended to be done after survival step)
    for (int i=0; i < hof.size(); ++i) 
    {
        // TODO: i guess the right way of doing this is using island indexes (or just take the hall of fame)
        std::cout << "index" << hof[i] << std::endl;
        const auto& ind = *pop.individuals.at(hof[i]);

        std::cout << ind.program.get_model() << std::endl;

        std::cout << "got individual of rank" << ind.fitness.rank << std::endl;
        if (val)
            f = ind.fitness.loss_v;
        else
            f = ind.fitness.loss;

        if (f < bs 
            || (f == bs && ind.fitness.complexity < this->best_complexity)
            )
        {
            std::cout << "updated" << std::endl;
        
            bs = f;
            this->best_ind = ind; 
            this->best_complexity = ind.fitness.complexity;

            updated = true;
        }
    }

    this->best_loss = bs; 

    return updated;
}


template <ProgramType T>
void Estimator<T>::run(Dataset &data)
{
    // It is up to the python side to create the dataset (we have a cool wrapper for that)
    std::cout << "starting to run" << std::endl;

    //TODO: i need to make sure i initialize everything (pybind needs to have constructors
    // without arguments to work, and i need to handle correcting these values before running)
    this->ss = SearchSpace(data, params.functions);
    std::cout << "search space was set" << std::endl;

    this->init();
    std::cout << "estimator initialized" << std::endl;

    pop.init(this->ss, this->params);

    std::cout << "pop initialized with size " << params.pop_size << " and " << params.num_islands << "islands" << std::endl;
    std::cout << pop.print_models() << std::endl;

    evaluator.set_scorer(params.scorer_);
    std::cout << "evaluator configured. starting to run " << std::endl;

    Dataset &batch = data;

    int threads;
    if (params.n_jobs == -1)
        threads = std::thread::hardware_concurrency();
    else if (params.n_jobs == 0)
        threads = params.num_islands;
    else
        threads = params.n_jobs;

    tf::Executor executor(threads); // TODO: executor could be an attribute (so I can move a lot of stuff here to init)
    std::cout << "using n threads " << threads << std::endl;

    assert( (executor.num_workers() > 0) && "Invalid number of workers");

    tf::Taskflow taskflow;

    // TODO: get references to all classes ( so they can be captured by taskflow) (like some private getters and setters)
    
    std::cout << "stop criteria is ready " << std::endl;
    // stop criteria 
    unsigned generation = 0;
    auto stop = [&]() {
        std::cout << "inside stop " << std::endl;
        return generation == params.gens; // TODO: max stall, max time, etc
    };

    // TODO: check that I dont use pop.size() (or I use correctly, because it will return the size with the slots for the offspring)
    // vectors to store each island separatedly
    vector<vector<size_t>> island_parents;
    vector<vector<size_t>> survivors;

    std::cout << "vectors are created " << std::endl;
    // TODO: progress bar? (it would be cool)
    // heavily inspired in https://github.com/heal-research/operon/blob/main/source/algorithms/nsga2.cpp
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&]() { /* done nothing to do */ }, // init (entry point for taskflow)

        stop, // loop condition
        
        [&](tf::Subflow& subflow) { // loop body (evolutionary main loop)
            std::cout << "inside body" << std::endl;
            auto prepare_gen = subflow.emplace([&]() { 
                std::cout << "inside prepare gen" << std::endl;
                std::cout << " -------------------- generation " << generation << " -------------------- " << std::endl;
                params.set_current_gen(generation);
                batch = data.get_batch(); // will return the original dataset if it is set to dont use batch 

                island_parents.clear();
                island_parents.resize(pop.num_islands);

                survivors.clear();
                survivors.resize(pop.num_islands);

                for (int i=0; i< params.num_islands; i++){
                    size_t idx_start = std::floor(i*params.pop_size/params.num_islands);
                    size_t idx_end   = std::floor((i+1)*params.pop_size/params.num_islands);

                    // auto delta = survivors.at(j).size(); // should have the same size as idx_end - idx_start
                    auto delta = idx_end - idx_start;

                    survivors.at(i).clear();
                    island_parents.at(i).clear();

                    survivors.at(i).resize(delta);
                    island_parents.at(i).resize(delta);
                }
            
                ++generation;
            }).name("prepare generation");// set generation in params, get batch

            auto select_parents = subflow.for_each_index(0, this->params.num_islands, 1, [&](int island) {
                std::cout << "inside select parents" << std::endl;
                evaluator.update_fitness(this->pop, island, data, params, true); // fit the weights with all training data

                // TODO: have some way to set which fitness to use (for example in params, or it can infer based on split size idk)
                // TODO: if using batch, fitness should be called before selection to set the batch
                if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
                    evaluator.update_fitness(this->pop, island, batch, params, false);

                vector<size_t> parents = selector.select(this->pop, island, params);

                for (int i=0; i< parents.size(); i++){
                    std::cout << i << std::endl;
                    island_parents.at(island).at(i) = parents.at(i);
                }
            }).name("select parents for each island");

            // this is not thread safe. But it is nice to keep out of parallel execution the bits of the
            // code that uses random generators (i think this helps to having random_seed to work properly). Also,
            // fit and evaluation are paralellized in survive_population, and these are expensive to run 
            auto generate_offspring = subflow.emplace([&]() {
                
                for (int island=0; island < params.num_islands; island++){
                    std::cout << "inside generate offspring" << std::endl;
                    this->pop.add_offspring_indexes(island); // we just need to add them, not remove (they are removed in survival step, that will return a selection with the same number of individuals as the original island size)
                    
                    std::cout << "before vary" << std::endl;
                    
                    // // variation to produce offspring
                    variator.vary(this->pop, island, island_parents.at(island));
                    std::cout << "before update fitness" << std::endl;
                }
            }).name("generate offspring for each island");
            
            auto survive_population = subflow.for_each_index(0, this->params.num_islands, 1, [&](int island) {
                
                evaluator.update_fitness(this->pop, island, data, params, true);
                // evaluator.validation(*this->pop, island_range, data, params);
                std::cout << "before batch update" << std::endl;

                if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
                    evaluator.update_fitness(this->pop, island, batch, params, false);
                std::cout << "before survive" << std::endl;

                // select survivors from combined pool of parents and offspring
                vector<size_t> island_survivors = survivor.survive(this->pop, island, params);
                std::cout << "before assign to survivors array" << std::endl;

                for (int i=0; i< island_survivors.size(); i++){
                    std::cout << i << std::endl;
                    survivors.at(island).at(i) = island_survivors.at(i);
                }
            }).name("evaluate offspring and select survivors");

            auto update_pop = subflow.emplace([&]() {
                std::cout << "before updating survivors" << std::endl;
                std::cout << pop.print_models() << std::endl;
                this->pop.update(survivors);
                
                std::cout << "after updating survivors" << std::endl;
                std::cout << pop.print_models() << std::endl;
            }).name("update population and detangle indexes");
            
            auto migration = subflow.emplace([&]() {
                std::cout << "before migrating" << std::endl;
                std::cout << pop.print_models() << std::endl;
                this->pop.migrate();
                
                std::cout << "after migrating" << std::endl;
                std::cout << pop.print_models() << std::endl;
                }).name("migration between islands");
            
            // TODO: update best, update log, increment generation counter (but not set in params)
            auto finish_gen = subflow.emplace([&]() { bool updated_best = this->update_best(data); }).name("update best, log, archive");

            // set-up subflow graph
            prepare_gen.precede(select_parents);
            select_parents.precede(generate_offspring);
            generate_offspring.precede(survive_population);
            survive_population.precede(update_pop);
            update_pop.precede(migration);
            migration.precede(finish_gen);
        },

        [&]() { return 0; }, // jump back to the next iteration

        [&]() { this->set_is_fitted(true); } // work done, report last gen and stop
    ); // evolutionary loop

    init.name("init");
    cond.name("termination");
    body.name("main loop");
    back.name("back");
    done.name("done");
    taskflow.name("island_gp");

    init.precede(cond);
    cond.precede(body, done);
    body.precede(back);
    back.precede(cond);

    std::cout << "taskflow configured " << std::endl;
    executor.run(taskflow);
    
    std::cout << "submitted jobs " << std::endl;

    executor.wait_for_all();
    std::cout << "finished " << std::endl;
}
}