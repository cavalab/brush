#include "engine.h"

#include <iostream>
#include <fstream>

namespace Brush{

using namespace Pop;
using namespace Sel;
using namespace Eval;
using namespace Var;
using namespace MAB;

/// @brief initialize Feat object for fitting.
template <ProgramType T>
void Engine<T>::init()
{
    r.set_seed(params.get_random_state());

    set_is_fitted(false);

    this->pop        = Population<T>();
    this->evaluator = Evaluation<T>();
    this->selector  = Selection<T>(params.sel, false);
    this->survivor  = Selection<T>(params.surv, true);

    this->archive.set_objectives(params.get_objectives());

    timer.Reset();

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
        auto indices = pop.island_indexes.at(island);
        pop_size += indices.size();
    }

    ArrayXf scores(pop_size);
    ArrayXf scores_v(pop_size);
    
    // TODO: change all size_t to unsigned?
    ArrayXi sizes(pop_size);
    ArrayXi complexities(pop_size); 

    float error_weight = Individual<T>::weightsMap[params.scorer];

    int index = 0;
    for (int island=0; island<params.num_islands; ++island)
    {
        auto indices = pop.island_indexes.at(island);
        for (unsigned int i=0; i<indices.size(); ++i)
        {
            const auto& p = this->pop.individuals.at(indices[i]);

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
    float    best_score_v   = this->best_ind.fitness.get_loss_v();
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
            << params.max_gens << " [" + bar + space + "]\n";
    else
        std::cout << std::fixed << "Time elapsed "<< timer 
            << "/" << params.max_time 
            << " seconds (Generation "<< params.current_gen+1 
            << ") [" + bar + space + "]\n";
        
    std::cout << std::fixed
              << "Best model on Val:" << best_ind.program.get_model() << "\n" 
              << "Train Loss (Med): " << stats.best_score.back() << " (" << stats.med_score.back() << ")\n"
              << "Val Loss (Med): " << stats.best_score_v.back() << " (" << stats.med_score_v.back() << ")\n"
              << "Median Size (Max): " << stats.med_size.back() << " (" << stats.max_size.back() << ")\n"
              << "Median complexity (Max): " << stats.med_complexity.back() << " (" << stats.max_complexity.back() << ")\n"
              << "Time (s): " << timer
              <<"\n\n";
}

template <ProgramType T>
vector<json> Engine<T>::get_archive_as_json()
{
    vector<json> archive_vector; 
    for (const auto& ind : archive.individuals) {
        json j;
        to_json(j, ind);
        archive_vector.push_back(j);
    }

    return archive_vector;
}

// TODO: dont have a get_pop and get_serialized_pop --> make them the same name but overloaded.
// Also fix this in the pybind wrapper
template <ProgramType T>
vector<json> Engine<T>::get_population_as_json()
{
    vector<json> pop_vector;
    for (const auto& ind : pop.individuals) {
        if (ind == nullptr) {
            // HANDLE_ERROR_THROW("get_population found a nullptr individual");
            continue;
        }

        json j;
        to_json(j, *ind);
        pop_vector.push_back(j);
    }

    if(pop_vector.size() != params.pop_size)
        HANDLE_ERROR_THROW("Population size is different from pop_size");

    return pop_vector;
}

template <ProgramType T>
vector<Individual<T>> Engine<T>::get_archive()
{
    vector<Individual<T>> archive_vector;
    for (const auto& ind : archive.individuals) {
        archive_vector.push_back(ind);
    }

    return archive_vector;
}

template <ProgramType T>
vector<Individual<T>> Engine<T>::get_population()
{
    vector<Individual<T>> pop_vector; 
    for (const auto& ind : pop.individuals) {
        if (ind == nullptr) {
            continue;
        }
        pop_vector.push_back(*ind);
    }

    if(pop_vector.size() != params.pop_size)
        HANDLE_ERROR_THROW("Population size is different from pop_size");

    return pop_vector;
}

template <ProgramType T>
void Engine<T>::set_population_from_json(vector<json> pop_vector)
{
    vector<Individual<T>> new_pop;

    // load serialized individuals
    for (const auto& ind_j : pop_vector) {
        Individual<T> ind;

        // deserialize individual
        from_json(ind_j, ind);

        // set reference to search space
        ind.program.set_search_space(ss);

        new_pop.push_back(ind);
    }

    // check if size matches
    if(new_pop.size() != params.pop_size)
        HANDLE_ERROR_THROW("set_population size is different from params.pop_size");

    // re-initialize population
    this->pop.init(new_pop, params);
}

template <ProgramType T>
void Engine<T>::set_population(vector<Individual<T>> pop_vector)
{
    vector<Individual<T>> new_pop;
    for (const auto& ind_j : pop_vector) {
        Individual<T> ind;
        ind.program = ind_j.program;
        ind.set_objectives(ind_j.get_objectives());

        new_pop.push_back(ind);
    }

    if(new_pop.size() != params.pop_size)
        HANDLE_ERROR_THROW("set_population size is different from params.pop_size");

    this->pop.init(new_pop, params);
}


template <ProgramType T>
void Engine<T>::lock_nodes(int end_depth, bool keep_leaves_unlocked)
{
    // iterate over the population, locking the program's tree nodes
    for (int island=0; island<pop.num_islands; ++island) {
        auto indices = pop.get_island_indexes(island);

        for (unsigned i = 0; i<indices.size(); ++i)
        {
            const auto& ind = pop.individuals.at(indices.at(i));
            ind->program.lock_nodes(end_depth, keep_leaves_unlocked);
        }
    }
}

template <ProgramType T>
bool Engine<T>::update_best()
{
    bool updated = false;
    bool passed;

    vector<size_t> merged_islands(0);
    for (int j=0;j<pop.num_islands; ++j)
    {
        auto indices = pop.island_indexes.at(j);
        for (int i=0; i<indices.size(); ++i)
        {
            merged_islands.push_back(indices.at(i));
        }
    }

    for (int i=0; i < merged_islands.size(); ++i) 
    {
        const auto& ind = *pop.individuals.at(merged_islands[i]);

        // TODO: use intermediary variables for wvalues
        // Iterate over the weighted values to compare (everything is a maximization problem here)
        passed = false;
        for (size_t j = 0; j < ind.fitness.get_wvalues().size(); ++j) {
            if (ind.fitness.get_wvalues()[j] > this->best_ind.fitness.get_wvalues()[j]) {
                passed = true;
                break;
            }
            if (ind.fitness.get_wvalues()[j] < this->best_ind.fitness.get_wvalues()[j]) {
                // it is not better, and it is also not equal. So, it is worse. Stop here.
                break;
            }
            // if no break, then its equal, so we keep going
        }

        if (passed)
        {
            this->best_ind = ind; 
            updated = true;
        }
    }

    return updated;
}


template <ProgramType T>
void Engine<T>::run(Dataset &data)
{
    // avoid re-initializing stuff so we can perform partial fits
    if (!this->is_fitted){
        //TODO: i need to make sure i initialize everything (pybind needs to have constructors
        // without arguments to work, and i need to handle correcting these values before running)

        // initializing classes that need data references    
        this->ss.init(data, params.functions, params.weights_init);    
        
        // TODO: make init to take necessary arguments and perform all initializations inside that function
        this->init();

        if (params.load_population != "") {
            this->pop.load(params.load_population);
        }
        else if (this->pop.individuals.size() == 0)
        {
            // This only works because the Population constructor resizes individuals to zero.
            this->pop.init(this->ss, this->params);
        }
    }
    
    // TODO: Should I refit them? or use the values at it is? (the fitness WILL BE recalculated regardless)
    // invalidating all individuals (so they are fitted with current data)
    // for (auto& individual : this->pop.individuals) {
    //     if (individual != nullptr) {
    //         // will force re-fit and calc all fitness information
    //         individual->set_is_fitted(false);
    //     }
    // }

    // This is data dependent so we initialize it everytime, regardless of partial fit
    // TODO: make variator have a default constructor and make it an attribute of engine
    Variation<T> variator = Variation<T>(this->params, this->ss, data);
    
    // log file stream
    std::ofstream log;
    if (!params.logfile.empty())
        log.open(params.logfile, std::ofstream::app);

    evaluator.set_scorer(params.scorer);

    Dataset &batch = data;

    int threads;
    if (params.n_jobs == -1)
        threads = std::thread::hardware_concurrency();
    else if (params.n_jobs == 0)
        threads = params.num_islands;
    else
        threads = params.n_jobs;

    tf::Executor executor(threads);

    assert( (executor.num_workers() > 0) && "Invalid number of workers");

    tf::Taskflow taskflow;

    // stop criteria 
    unsigned generation = 0;
    unsigned stall_count = 0;
    float fraction = 0;

    auto stop = [&]() {
        bool condition = ( (generation == params.max_gens)
               || (params.max_stall != 0 && stall_count > params.max_stall) 
               || (params.max_time != -1 && timer.Elapsed().count() > params.max_time)
        );

        return condition;
    };

    // TODO: check that I dont use pop.size() (or I use correctly, because it will return the size with the slots for the offspring)
    // vectors to store each island separatedly
    vector<vector<size_t>> island_parents;
    
    island_parents.clear();
    island_parents.resize(pop.num_islands);

    for (int i=0; i< params.num_islands; i++){
        size_t idx_start = std::floor(i*params.pop_size/params.num_islands);
        size_t idx_end   = std::floor((i+1)*params.pop_size/params.num_islands);

        auto delta = idx_end - idx_start;

        island_parents.at(i).clear();
        island_parents.at(i).resize(delta);
    }

    // heavily inspired in https://github.com/heal-research/operon/blob/main/source/algorithms/nsga2.cpp
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&](tf::Subflow& subflow) { 
            auto fit_init_pop = subflow.for_each_index(0, this->params.num_islands, 1, [&](int island) {
                // Evaluate the individuals at least once
                // Set validation loss before calling update best

                evaluator.update_fitness(this->pop, island, data, params, true, true);
            });
            auto find_init_best = subflow.emplace([&]() { 
                // Make sure we initialize it. We do this update here because we need to 
                // have the individuals fitted before we can compare them. When update_best
                // is called, we are garanteed that the individuals are fitted and have valid 
                // fitnesses.
                this->best_ind = *pop.individuals.at(0);
                this->update_best(); // at this moment we dont care about update_best return value
            });
            fit_init_pop.precede(find_init_best);
         }, // init (entry point for taskflow)

        stop, // loop condition
        
        [&](tf::Subflow& subflow) { // loop body (evolutionary main loop)
            auto prepare_gen = subflow.emplace([&]() { 
                params.set_current_gen(generation);
                batch = data.get_batch(); // will return the original dataset if it is set to dont use batch 
            }).name("prepare generation");// set generation in params, get batch

            auto run_generation = subflow.for_each_index(0, this->params.num_islands, 1, [&](int island) {

                evaluator.update_fitness(this->pop, island, data, params, false, false); // fit the weights with all training data
                
                // TODO: have some way to set which fitness to use (for example in params, or it can infer based on split size idk)
                // TODO: if using batch, fitness should be called before selection to set the batch
                if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
                    evaluator.update_fitness(this->pop, island, batch, params, false, false);

                vector<size_t> parents = selector.select(this->pop, island, params);
                for (int i=0; i< parents.size(); i++){
                    island_parents.at(island).at(i) = parents.at(i);
                }
                
                this->pop.add_offspring_indexes(island); 

            }).name("runs one generation at each island in parallel");

            auto update_pop = subflow.emplace([&]() { // sync point
                // Variation is not thread safe.
                // TODO: optimize this and make it work with multiple islands in parallel.
                for (int island = 0; island < this->params.num_islands; ++island) {

                    // TODO: do I have to pass data as an argument here? or can I use the instance reference
                    variator.vary_and_update(this->pop, island, island_parents.at(island),
                                             data, evaluator, 
                                             
                                             // conditions to apply simplification.
                                             // It starts only on the second half of generations,
                                             // and it is not applied every generation. 
                                             // Also, we garantee that the final generation
                                             // will be simplified.
                                             (generation>=params.max_gens/2) || (stall_count == params.max_stall-1) 
                                            );
                }

                // select survivors from combined pool of parents and offspring.
                // if the same individual exists in different islands, then it will be selected several times and the pareto front will have less diversity.
                // to avoid this, survive should be unified
                // TODO: survivor should still take params?
                // TODO: RETURN SINGLE VECTOR and stop wrapping it into a single-element vector

                auto survivor_indices = survivor.survive(this->pop, 0, params);

                // TODO: do i need these next this-> pointers?
                variator.update_ss();
                this->pop.update({survivor_indices});
                this->pop.migrate();
            }).name("update, migrate and disentangle indexes between islands");
            
            auto finish_gen = subflow.emplace([&]() {
                // Set validation loss before calling update best
                for (int island = 0; island < this->params.num_islands; ++island) {
                    evaluator.update_fitness(this->pop, island, data, params, false, true);
                }

                archive.update(pop, params);

                bool updated_best = this->update_best();
                
                fraction = params.max_time == -1 ? ((generation+1)*1.0)/params.max_gens : 
                                                    timer.Elapsed().count()/params.max_time;

                if ( params.verbosity>1 || !params.logfile.empty()) {
                    calculate_stats();
                }

                if(params.verbosity>1)
                {
                    print_stats(log, fraction);
                }
                else if(params.verbosity == 1)
                    print_progress(fraction);

                if (!params.logfile.empty())
                    log_stats(log);
                    
                if (generation == 0 || updated_best )
                    stall_count = 0;
                else
                    ++stall_count;
                
                ++generation;

            }).name("update best, update ss, log, archive, stall");

            // set-up subflow graph
            prepare_gen.precede(run_generation);
            run_generation.precede(update_pop);
            update_pop.precede(finish_gen);
        },

        [&]() { return 0; }, // jump back to the next iteration

        [&](tf::Subflow& subflow) {
            // set VALIDATION loss for archive
            for (int island = 0; island < this->params.num_islands; ++island) {
                evaluator.update_fitness(this->pop, island, data, params, true, true);
            }

            calculate_stats();
            archive.update(pop, params);

            if (params.save_population != "")
                this->pop.save(params.save_population);

            set_is_fitted(true);
            
            if (!params.logfile.empty()) {
                std::ofstream log_simplification;
                log_simplification.open(params.logfile+"simplification_table", std::ofstream::app);
                variator.log_simplification_table(log_simplification);
                
                log_simplification.close();
            }

            // TODO: open, write, close? (to avoid breaking the file and allow some debugging if things dont work well)
            if (log.is_open())
                log.close();

            // getting the updated versions
            this->ss = variator.search_space;
            this->params = variator.parameters;

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

    executor.run(taskflow);
    executor.wait_for_all();
    
    //When you have tasks that are created at runtime (e.g., subflow,
    // cudaFlow), you need to execute the graph first to spawn these tasks and dump the entire graph.
}
}