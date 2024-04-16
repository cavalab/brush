#include "engine.h"


#include <iostream>
#include <fstream>


namespace Brush{


using namespace Pop;
using namespace Sel;
using namespace Eval;
using namespace Var;

/// @brief initialize Feat object for fitting.
template <ProgramType T>
void Engine<T>::init()
{
    // std::cout << "inside init" << std::endl;

    // TODO: initialize (set operator) for survivor and selector
    // initialize population with initial model and/or starting pop

    // TODO: get rid of omp
    if (params.n_jobs!=0) 
        omp_set_num_threads(params.get_n_jobs());

    // std::cout << "set number of threads" << std::endl;

    r.set_seed(params.get_random_state());
    // std::cout << "set random state" << std::endl;

    // set up the pop, variator, etc
    set_is_fitted(false);
    // std::cout << "is fitted is false" << std::endl;

   this->pop = Population<T>();
    //std::cout << "created population" << std::endl;

    // TODO: load population into file
    // TODO: if initializing from a population file, then this is where we should load previous models.
    // three behaviors: if we have only 1 ind, then replicate it trought the entire pop
    // if n_ind is the same as pop_size, load all models. if n_ind != pop_size, throw error
    if (params.load_population != "")
        this->pop.load(params.load_population);

    this->evaluator = Evaluation<T>();
    //std::cout << "created evaluator" << std::endl;

    // TOD: make these classes have a default constructor, and stop recreating instances
    this->variator.init(params, ss);
    //std::cout << "initialized variator" << std::endl;

    this->selector = Selection<T>(params.sel, false);
    //std::cout << "created selector" << std::endl;

    this->survivor = Selection<T>(params.surv, true);
    //std::cout << "created survivor" << std::endl;

    this->best_loss = MAX_FLT;
    this->best_complexity = MAX_FLT;

    // TODO getters and setters for the best solution found after evolution
    // predict, transform, predict_proba, etc.
    // get statistics
    // load and save best individuals
    // logger, save to file
    // execution archive
    // score functions
    // fit methods (this will run the evolution)

    // start the clock
    timer.Reset();

    // // signal handler
    // signal(SIGINT, my_handler);

    // // reset statistics
    // this->stats = Log_Stats();
}

template <ProgramType T> // TODO: use the dataset, or ignore it
bool Engine<T>::update_best(const Dataset& data, bool val)
{
    //std::cout << "updating best" << std::endl;

    float bs;
    bs = this->best_loss; 
    
    float f; 
    // TODO: archive here?

    bool updated = false; 

    //std::cout << "inside loop" << std::endl;

    vector<size_t> hof = this->pop.hall_of_fame(1);

    //std::cout << "got hof" << std::endl;

    for (int i=0; i < hof.size(); ++i) 
    {
        //std::cout << "index" << hof[i] << std::endl;
        const auto& ind = *pop.individuals.at(hof[i]);

        //std::cout << ind.program.get_model() << std::endl;

        //std::cout << "got individual of rank" << ind.fitness.rank << std::endl;
        if (val)
            f = ind.fitness.loss_v;
        else
            f = ind.fitness.loss;

        if (f < bs 
            || (f == bs && ind.fitness.complexity < this->best_complexity)
            )
        {
            //std::cout << "updated" << std::endl;
        
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
void Engine<T>::run(Dataset &data)
{
    // It is up to the python side to create the dataset (we have a cool wrapper for that)
    //std::cout << "starting to run" << std::endl;

    //TODO: i need to make sure i initialize everything (pybind needs to have constructors
    // without arguments to work, and i need to handle correcting these values before running)
    this->ss = SearchSpace(data, params.functions);
    //std::cout << "search space was set" << std::endl;

    this->init();
    //std::cout << "Engine initialized" << std::endl;

    pop.init(this->ss, this->params);

    //std::cout << "pop initialized with size " << params.pop_size << " and " << params.num_islands << "islands" << std::endl;
    //std::cout << pop.print_models() << std::endl;

    evaluator.set_scorer(params.scorer_);
    //std::cout << "evaluator configured. starting to run " << std::endl;

    Dataset &batch = data;

    int threads;
    if (params.n_jobs == -1)
        threads = std::thread::hardware_concurrency();
    else if (params.n_jobs == 0)
        threads = params.num_islands;
    else
        threads = params.n_jobs;

    tf::Executor executor(threads); // TODO: executor could be an attribute (so I can move a lot of stuff here to init)
    //std::cout << "using n threads " << threads << std::endl;

    assert( (executor.num_workers() > 0) && "Invalid number of workers");

    tf::Taskflow taskflow;

    // TODO: get references to all classes ( so they can be captured by taskflow) (like some private getters and setters)
    
    //std::cout << "stop criteria is ready " << std::endl;
    // stop criteria 
    unsigned generation = 0;
    unsigned stall_count = 0;
    float fraction = 0;

    auto stop = [&]() {
        //std::cout << "inside stop " << std::endl;
        // TODO: max time
        return (  (generation == params.gens)
               && (params.max_stall == 0 || stall_count < params.max_stall) 
               && (params.max_time == -1 || params.max_time > timer.Elapsed().count())
        );
    };

    // TODO: check that I dont use pop.size() (or I use correctly, because it will return the size with the slots for the offspring)
    // vectors to store each island separatedly
    vector<vector<size_t>> island_parents;
    vector<vector<size_t>> survivors;
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

    //std::cout << "vectors are created " << std::endl;
    // TODO: progress bar? (it would be cool)
    // heavily inspired in https://github.com/heal-research/operon/blob/main/source/algorithms/nsga2.cpp
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&]() { /* done nothing to do */ }, // init (entry point for taskflow)

        stop, // loop condition
        
        [&](tf::Subflow& subflow) { // loop body (evolutionary main loop)
            //std::cout << "inside body" << std::endl;
            auto prepare_gen = subflow.emplace([&]() { 
                //std::cout << "inside prepare gen" << std::endl;
                //std::cout << " -------------------- generation " << generation << " -------------------- " << std::endl;
                params.set_current_gen(generation);
                batch = data.get_batch(); // will return the original dataset if it is set to dont use batch 

                // island_parents.clear();
                // island_parents.resize(pop.num_islands);

                // survivors.clear();
                // survivors.resize(pop.num_islands);

                // for (int i=0; i< params.num_islands; i++){
                //     size_t idx_start = std::floor(i*params.pop_size/params.num_islands);
                //     size_t idx_end   = std::floor((i+1)*params.pop_size/params.num_islands);

                //     // auto delta = survivors.at(j).size(); // should have the same size as idx_end - idx_start
                //     auto delta = idx_end - idx_start;

                //     survivors.at(i).clear();
                //     island_parents.at(i).clear();

                //     survivors.at(i).resize(delta);
                //     island_parents.at(i).resize(delta);
                // }
            
            }).name("prepare generation");// set generation in params, get batch

            auto run_generation = subflow.for_each_index(0, this->params.num_islands, 1, [&](int island) {
                //std::cout << "inside select parents" << std::endl;
                evaluator.update_fitness(this->pop, island, data, params, true); // fit the weights with all training data

                // TODO: individuals should have a flag is_fitted so we avoid re-fitting them

                // TODO: have some way to set which fitness to use (for example in params, or it can infer based on split size idk)
                // TODO: if using batch, fitness should be called before selection to set the batch
                if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
                    evaluator.update_fitness(this->pop, island, batch, params, false);

                vector<size_t> parents = selector.select(this->pop, island, params);

                for (int i=0; i< parents.size(); i++){
                    //std::cout << i << std::endl;
                    island_parents.at(island).at(i) = parents.at(i);
                }
                
                //std::cout << "inside generate offspring" << std::endl;
                this->pop.add_offspring_indexes(island); 

                //std::cout << "before vary" << std::endl;
                // // variation to produce offspring
                variator.vary(this->pop, island, island_parents.at(island));
                //std::cout << "before update fitness" << std::endl;

                evaluator.update_fitness(this->pop, island, data, params, true);
                // evaluator.validation(*this->pop, island_range, data, params);
                //std::cout << "before batch update" << std::endl;

                if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
                    evaluator.update_fitness(this->pop, island, batch, params, false);
                //std::cout << "before survive" << std::endl;

                // select survivors from combined pool of parents and offspring
                vector<size_t> island_survivors = survivor.survive(this->pop, island, params);
                //std::cout << "before assign to survivors array" << std::endl;

                for (int i=0; i< island_survivors.size(); i++){
                    //std::cout << i << std::endl;
                    survivors.at(island).at(i) = island_survivors.at(i);
                }
            }).name("runs one generation at each island in parallel");

            auto update_pop = subflow.emplace([&]() {
                //std::cout << "before updating survivors" << std::endl;
                //std::cout << pop.print_models() << std::endl;
                this->pop.update(survivors);
                
                //std::cout << "after updating survivors" << std::endl;
                //std::cout << pop.print_models() << std::endl;
                
                //std::cout << "before migrating" << std::endl;
                //std::cout << pop.print_models() << std::endl;
                this->pop.migrate();
                
                //std::cout << "after migrating" << std::endl;
                //std::cout << pop.print_models() << std::endl;
            }).name("update, migrate and disentangle indexes between islands");
            
            // TODO: update log and archive
            auto finish_gen = subflow.emplace([&]() {
                bool updated_best = this->update_best(data);
                
                if (generation == 0 || updated_best )
                    stall_count = 0;
                else
                    ++stall_count;
                
                ++generation;

            }).name("update best, log, archive, stall");

            // set-up subflow graph
            prepare_gen.precede(run_generation);
            run_generation.precede(update_pop);
            update_pop.precede(finish_gen);
        },

        [&]() { return 0; }, // jump back to the next iteration

        [&]() {
            if (params.save_population != "")
                this->pop.save(params.save_population);

            this->set_is_fitted(true);
        } // work done, report last gen and stop
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

    //std::cout << "taskflow configured " << std::endl;
    executor.run(taskflow);
    
    //std::cout << "submitted jobs " << std::endl;

    executor.wait_for_all();
    //std::cout << "finished " << std::endl;
    
    //When you have tasks that are created at runtime (e.g., subflow,
    // cudaFlow), you need to execute the graph first to spawn these tasks and dump the entire graph.

    //std::cout << "dumping taskflow in json " << std::endl;
    taskflow.dump(std::cout); 
}
}