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

    // initializing survivor and selector based on params
    this->selector = Selection<T>(params.sel, false);
    this->survivor = Selection<T>(params.surv, true);

    this->best_score = MAX_FLT;
    this->best_complexity = MAX_FLT;

    // TODO: predict, transform, predict_proba, fit (will run the engine)

    this->archive.set_objectives(params.objectives);

    // start the clock
    timer.Reset();

    // // signal handler
    // signal(SIGINT, my_handler);

    // reset statistics
    this->stats = Log_Stats();
}

template <ProgramType T>
void Engine<T>::print_progress(float percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;

    printf ("\rCompleted %3d%% [%.*s%*s]", val, lpad, PBSTR.c_str(), rpad, "");
    
    fflush (stdout);
    
    if(val == 100)
        cout << "\n";
}


template <ProgramType T>
void Engine<T>::calculate_stats()
{
    int pop_size = 0;
    for (int island=0; island<params.num_islands; ++island)
    {
        auto idxs = pop.island_indexes.at(island);
        pop_size += idxs.size();
    }

    ArrayXf scores(pop_size);
    ArrayXf scores_v(pop_size);
    
    // TODO: change all size_t to unsigned?
    ArrayXi sizes(pop_size);
    ArrayXi complexities(pop_size); 

    float error_weight = Individual<T>::weightsMap[params.scorer_];

    int index = 0;
    for (int island=0; island<params.num_islands; ++island)
    {
        auto idxs = pop.island_indexes.at(island);
        for (unsigned int i=0; i<idxs.size(); ++i)
        {
            const auto& p = this->pop.individuals.at(idxs[i]);

            // Fitness class will store every information that can be used as
            // fitness. you just need to access them. Multiplying by weight
            // so we can find best score. From Fitness::dominates:
            //     the proper way of comparing weighted values is considering
            //     everything as a maximization problem
            scores(index)       = p->fitness.get_loss();
            scores_v(index)     = p->fitness.get_loss_v();
            sizes(index)        = p->get_size(); 
            complexities(index) = p->get_complexity(); 
            ++index;
        }
    }

    assert (pop_size == this->params.pop_size);

    // Multiply by weight to make it a maximization problem.
    // Then, multiply again to get rid of signal
    float    best_score     = (scores*error_weight).maxCoeff()*error_weight;
    float    best_score_v   = (scores_v*error_weight).maxCoeff()*error_weight;
    float    med_score      = median(scores); 
    float    med_score_v    = median(scores_v); 
    unsigned med_size       = median(sizes);                        
    unsigned med_complexity = median(complexities);
    unsigned max_size       = sizes.maxCoeff();
    unsigned max_complexity = complexities.maxCoeff();
    
    // update stats
    stats.update(params.current_gen,
                 timer.Elapsed().count(),
                 best_score,
                 best_score_v,
                 med_score,
                 med_score_v,
                 med_size,
                 med_complexity,
                 max_size,
                 max_complexity);
}


template <ProgramType T>
void Engine<T>::log_stats(std::ofstream& log)
{
    // print stats in tabular format
    string sep = ",";
    if (params.current_gen == 0) // print header
    {
        log << "generation"     << sep
            << "time"           << sep
            << "best_score"     << sep 
            << "best_score_val" << sep 
            << "med_score"      << sep 
            << "med_score_val"  << sep 
            << "med_size"       << sep 
            << "med_complexity" << sep 
            << "max_size"       << sep 
            << "max_complexity" << "\n";
    }
    log << params.current_gen          << sep
        << timer.Elapsed().count()     << sep
        << stats.best_score.back()     << sep
        << stats.best_score_v.back()   << sep
        << stats.med_score.back()      << sep
        << stats.med_score_v.back()    << sep
        << stats.med_size.back()       << sep
        << stats.med_complexity.back() << sep
        << stats.max_size.back()       << sep
        << stats.max_complexity.back() << "\n"; 
}

template <ProgramType T>
void Engine<T>::print_stats(std::ofstream& log, float fraction)
{
    // progress bar
    string bar, space = "";                                 
    for (unsigned int i = 0; i<50; ++i)
    {
        if (i <= 50*fraction) bar += "/";
        else space += " ";
    }

    std::cout.precision(5);
    std::cout << std::scientific;
    
    if(params.max_time == -1)
        std::cout << "Generation " << params.current_gen+1 << "/" 
            << params.gens << " [" + bar + space + "]\n";
    else
        std::cout << std::fixed << "Time elapsed "<< timer 
            << "/" << params.max_time 
            << " seconds (Generation "<< params.current_gen+1 
            << ") [" + bar + space + "]\n";
        
    std::cout << std::fixed
              << "Train Loss (Med): " << stats.best_score.back() << " (" << stats.med_score.back() << ")\n"
              << "Val Loss (Med): " << stats.best_score_v.back() << " (" << stats.med_score_v.back() << ")\n"
              << "Median Size (Max): " << stats.med_size.back() << " (" << stats.max_size.back() << ")\n"
              << "Median complexity (Max): " << stats.med_complexity.back() << " (" << stats.max_complexity.back() << ")\n"
              << "Time (s): " << timer
              <<"\n\n";
}


template <ProgramType T> // TODO: use the dataset, or ignore it
bool Engine<T>::update_best(const Dataset& data, bool val)
{
    //std::cout << "updating best" << std::endl;

    float error_weight = Individual<T>::weightsMap[params.scorer_];

    float bs;
    bs = this->best_score; 
    
    float f;

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

        if (f*error_weight > bs*error_weight
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

    this->best_score = bs; 

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

    // log file stream
    std::ofstream log;
    if (!params.logfile.empty())
        log.open(params.logfile, std::ofstream::app);

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

    //std::cout << "stop criteria is ready " << std::endl;
    // stop criteria 
    unsigned generation = 0;
    unsigned stall_count = 0;
    float fraction = 0;

    auto stop = [&]() {
        return (  (generation == params.gens)
               && ((params.max_stall == 0 || stall_count < params.max_stall) 
               &&  (params.max_time == -1 || params.max_time > timer.Elapsed().count()) )
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
            
            auto finish_gen = subflow.emplace([&]() {
                bool updated_best = this->update_best(data);
                
                // TODO: use_arch
                if ( params.verbosity>1 || !logfile.empty()) {
                    calculate_stats();
                }

                // TODO: logger working
                // logger.log("calculate stats...",2);

                // if (use_arch)  // TODO: archive
                //     archive.update(pop,params);
                
                fraction = params.max_time == -1 ? ((generation+1)*1.0)/params.gens : 
                                                    timer.Elapsed().count()/params.max_time;

                if(params.verbosity>1)
                    print_stats(log, fraction);    
                else if(params.verbosity == 1)
                    print_progress(fraction);

                if (!params.logfile.empty())
                    log_stats(log);
                    
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
            
            // TODO: make this work
            // if (save_pop > 0)
            // {
            //     pop.save(this->logfile+".pop.gen" + to_string(params.current_gen) 
            //             + ".json");
            //     this->best_ind.save(this->logfile+".best.json");
            // }
            
            // TODO: open, write, close? (to avoid breaking the file and allow some debugging if things dont work well)
            if (log.is_open())
                log.close();
                
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