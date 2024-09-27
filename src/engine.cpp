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

   this->pop = Population<T>();

    this->evaluator = Evaluation<T>();

    // TODO: make these classes have a default constructor, and stop recreating instances
    this->variator = Variation<T>(this->params, this->ss);

    this->selector = Selection<T>(params.sel, false);
    this->survivor = Selection<T>(params.surv, true);

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
            << params.max_gens << " [" + bar + space + "]\n";
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

template <ProgramType T>
vector<json> Engine<T>::get_archive(bool front)
{
    vector<json> archive_vector; // Use a vector to store serialized individuals
    
    // TODO: use this front argument (or remove it). I think I can remove 
    for (const auto& ind : archive.individuals) {
        json j;  // Serialize each individual
        to_json(j, ind);
        archive_vector.push_back(j);
    }

    return archive_vector;
}

// TODO: private function called find_individual that searches for it based on id. Then,
// use this function in predict_archive and predict_proba_archive.
template <ProgramType T>
auto Engine<T>::predict_archive(int id, const Dataset& data)
{
    if (id == best_ind.id)
        return best_ind.predict(data);

    for (int i = 0; i < this->archive.individuals.size(); ++i)
    {
        Individual<T>& ind = this->archive.individuals.at(i);

        if (id == ind.id)
            return ind.predict(data);
    }
    for (int island=0; island<pop.num_islands; ++island) {
        auto indices = pop.get_island_indexes(island);

        for (unsigned i = 0; i<indices.size(); ++i)
        {
            const auto& ind = pop.individuals.at(indices.at(i));

            if (id == ind->id)
                return ind->predict(data);
        } 
    }

    std::runtime_error("Could not find id = "
            + to_string(id) + "in archive or population.");
        
    return best_ind.predict(data);
}

template <ProgramType T>
auto Engine<T>::predict_archive(int id, const Ref<const ArrayXXf>& X)
{
    Dataset d(X);
    return predict_archive(id, d);
}

template <ProgramType T>
template <ProgramType P>
    requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
auto Engine<T>::predict_proba_archive(int id, const Dataset& data)
{
    if (id == best_ind.id)
        return best_ind.predict_proba(data);

    for (int i = 0; i < this->archive.individuals.size(); ++i)
    {
        Individual<T>& ind = this->archive.individuals.at(i);

        if (id == ind.id)
            return ind.predict_proba(data);
    }
    for (int island=0; island<pop.num_islands; ++island) {
        auto indices = pop.get_island_indexes(island);

        for (unsigned i = 0; i<indices.size(); ++i)
        {
            const auto& ind = pop.individuals.at(indices.at(i));

            if (id == ind->id)
                return ind->predict_proba(data);
        } 
    }
     
    std::runtime_error("Could not find id = "
            + to_string(id) + "in archive or population.");
            
    return best_ind.predict_proba(data);
}

template <ProgramType T>
template <ProgramType P>
    requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
auto Engine<T>::predict_proba_archive(int id, const Ref<const ArrayXXf>& X)
{
    Dataset d(X);
    return predict_proba_archive(id, d);
}

template <ProgramType T>
bool Engine<T>::update_best()
{
    bool updated = false;
    bool passed;

    vector<size_t> hof = this->pop.hall_of_fame(1);

    for (int i=0; i < hof.size(); ++i) 
    {
        const auto& ind = *pop.individuals.at(hof[i]);

        // TODO: use intermediary variables for wvalues
        // Iterate over the weighted values to compare (everything is a maximization problem here)
        passed = false;
        for (size_t j = 0; j < this->best_ind.fitness.get_wvalues().size(); ++j) {
            if (ind.fitness.get_wvalues()[j] > this->best_ind.fitness.get_wvalues()[j]) {
                passed = true;
            }
            else if (ind.fitness.get_wvalues()[j] < this->best_ind.fitness.get_wvalues()[j]) {
                // it is not better, and it is also not equal. So, it is worse. Stop here.
                break;
            }
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
    //TODO: i need to make sure i initialize everything (pybind needs to have constructors
    // without arguments to work, and i need to handle correcting these values before running)
    
    this->ss.init(data, params.functions, params.weights_init);
    this->init();

    if (params.load_population != "") {
        this->pop.load(params.load_population);

        // invalidating all individuals
        for (auto& individual : this->pop.individuals) {
            if (individual != nullptr) {
                individual->set_is_fitted(false);
            }
        }
    }
    else
        this->pop.init(this->ss, this->params);

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
        return (  (generation == params.max_gens)
               || (params.max_stall != 0 && stall_count > params.max_stall) 
               || (params.max_time != -1 && timer.Elapsed().count() > params.max_time)
        );
    };

    // TODO: check that I dont use pop.size() (or I use correctly, because it will return the size with the slots for the offspring)
    // vectors to store each island separatedly
    vector<vector<size_t>> island_parents;
    vector<vector<size_t>> survivors;
    vector<vector<float>> rewards;
    
    island_parents.clear();
    island_parents.resize(pop.num_islands);

    survivors.clear();
    survivors.resize(pop.num_islands);

    rewards.clear();
    rewards.resize(pop.num_islands);

    for (int i=0; i< params.num_islands; i++){
        size_t idx_start = std::floor(i*params.pop_size/params.num_islands);
        size_t idx_end   = std::floor((i+1)*params.pop_size/params.num_islands);

        auto delta = idx_end - idx_start;

        survivors.at(i).clear();
        island_parents.at(i).clear();
        rewards.at(i).clear();

        survivors.at(i).resize(delta);
        island_parents.at(i).resize(delta);
        rewards.at(i).resize(delta);
    }

    // heavily inspired in https://github.com/heal-research/operon/blob/main/source/algorithms/nsga2.cpp
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&]() { /* done nothing to do */ }, // init (entry point for taskflow)

        stop, // loop condition
        
        [&](tf::Subflow& subflow) { // loop body (evolutionary main loop)
            auto prepare_gen = subflow.emplace([&]() { 
                params.set_current_gen(generation);
                batch = data.get_batch(); // will return the original dataset if it is set to dont use batch 
            }).name("prepare generation");// set generation in params, get batch

            auto run_generation = subflow.for_each_index(0, this->params.num_islands, 1, [&](int island) {
                evaluator.update_fitness(this->pop, island, data, params, true, false); // fit the weights with all training data

                // TODO: have some way to set which fitness to use (for example in params, or it can infer based on split size idk)
                // TODO: if using batch, fitness should be called before selection to set the batch
                if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
                    evaluator.update_fitness(this->pop, island, batch, params, false, false);

                vector<size_t> parents = selector.select(this->pop, island, params);

                for (int i=0; i< parents.size(); i++){
                    island_parents.at(island).at(i) = parents.at(i);
                }
                
                this->pop.add_offspring_indexes(island); 

                // TODO: optimize this and make it work with multiple islands in parallel.
                variator.vary_and_update(this->pop, island, island_parents.at(island),
                                         data, evaluator);

                // variator.vary(this->pop, island, island_parents.at(island));
                
                // evaluator.update_fitness(this->pop, island, data, params, true, false);

                // vector<float> island_rewards = variator.calculate_rewards(this->pop, island);
                // for (int i=0; i< island_rewards.size(); i++){
                //     rewards.at(island).at(i) = island_rewards.at(i);
                // }

                if (data.use_batch) // assign the batch error as fitness (but fit was done with training data)
                    evaluator.update_fitness(this->pop, island, batch, params, false, false);

                // select survivors from combined pool of parents and offspring
                vector<size_t> island_survivors = survivor.survive(this->pop, island, params);
                for (int i=0; i< island_survivors.size(); i++){
                    survivors.at(island).at(i) = island_survivors.at(i);
                }

            }).name("runs one generation at each island in parallel");

            auto update_pop = subflow.emplace([&]() { // sync point
                // Flatten the rewards vector

                // TODO: stop using flattened_rewards
                // TODO: improve vary_and_update and stop using this (as well as island_rewards in the subflow above)
                // vector<float> flattened_rewards;
                // for (const auto& island_rewards : rewards) {
                //     flattened_rewards.insert(flattened_rewards.end(),
                //                              island_rewards.begin(),
                //                              island_rewards.end());
                // }
                
                // // Assert that flattened_rewards has the same size as popsize
                // assert(flattened_rewards.size() == this->pop.size());
                
                // // TODO: do i need these next this-> pointers?
                // // Use the flattened rewards vector for updating the population
                
                this->variator.update_ss();
                this->pop.update(survivors);
                this->pop.migrate();
            }).name("update, migrate and disentangle indexes between islands");
            
            auto finish_gen = subflow.emplace([&]() {
                // TODO: figure out a better way to initialize best individual
                if (generation == 0)
                    this->best_ind = *pop.individuals.at(0);

                if ( (params.verbosity>1 || !params.logfile.empty() )
                || params.use_arch ) {
                    calculate_stats();
                }

                if (params.use_arch) {
                    archive.update(pop, params);
                }
                
                subflow.for_each_index(0, this->params.num_islands, 1, [&](int island) {
                    evaluator.update_fitness(this->pop, island, data, params, false, true);
                }).name("Set validation loss to update best");

                bool updated_best = this->update_best();
                
                fraction = params.max_time == -1 ? ((generation+1)*1.0)/params.max_gens : 
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

            }).name("update best, update ss, log, archive, stall");

            // set-up subflow graph
            prepare_gen.precede(run_generation);
            run_generation.precede(update_pop);
            update_pop.precede(finish_gen);
        },

        [&]() { return 0; }, // jump back to the next iteration

        [&](tf::Subflow& subflow) {            
            subflow.for_each_index(0, this->params.num_islands, 1, [&](int island) {
                evaluator.update_fitness(this->pop, island, data, params, false, false);
            }).name("Set train loss as fitness");

            // std::cout << "Variator probs:" << std::endl;
            // std::cout << "cx: " << variator.parameters.get_cx_prob() << std::endl;
            // for (const auto& [name, prob] : variator.parameters.get_mutation_probs())
            //     std::cout << name << ": " << prob << std::endl;

            // std::cout << "variator Search Space:" << std::endl;
            // variator.search_space.print();
        
            // std::cout << "Engine probs:" << std::endl;
            // std::cout << "cx: " << params.get_cx_prob() << std::endl;
            // for (const auto& [name, prob] : params.get_mutation_probs())
            //     std::cout << name << ": " << prob << std::endl;

            // std::cout << "Engine Search Space:" << std::endl;
            // ss.print();

            if (params.save_population != "")
                this->pop.save(params.save_population);

            set_is_fitted(true);
            
            // TODO: open, write, close? (to avoid breaking the file and allow some debugging if things dont work well)
            if (log.is_open())
                log.close();

            // if we're not using an archive, let's store the final population in the 
            // archive
            if (!params.use_arch)
            {
                //TODO: make this a function (use update from archive instead of repeating it here)
                std::cout << "saving final population as archive..." << std::endl;

                archive.individuals.resize(0);
                for (int island =0; island< pop.num_islands; ++island) {
                    vector<size_t> indices = pop.get_island_indexes(island);

                    for (unsigned i = 0; i<indices.size(); ++i)
                    {
                        archive.individuals.push_back( *pop.individuals.at(indices.at(i)) );
                    }
                }

                // removing duplicates
                std::sort(archive.individuals.begin(), archive.individuals.end(), 
                          &archive.sortObj1);

                auto it = std::unique(archive.individuals.begin(),archive.individuals.end(), 
                                      &archive.sameObjectives);

                archive.individuals.resize(std::distance(archive.individuals.begin(),it));
            }

            // getting the updated versions
            // TODO: make the probabilities add up to 1 (this doesnt matter for the cpp side, but it is a good practice and helps comparing different probabilities)
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