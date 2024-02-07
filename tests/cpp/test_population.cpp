#include "testsHeader.h"

#include "../../src/individual.cpp"
#include "../../src/population.cpp" // TODO: figure out if thats ok to include cpps instead of headers
#include "../../src/eval/evaluation.cpp"
#include "../../src/selection/nsga2.cpp"
#include "../../src/selection/selection_operator.cpp"
#include "../../src/selection/selection.cpp"

using namespace Brush::Pop;
using namespace Brush::Sel;
using namespace Brush::Eval;
using namespace Brush::Sel;

TEST(Population, PopulationTests)
{    
    // works with even and uneven pop sizes. (TODO: PARAMETERIZE this test to do it with even and uneven, and single individual pop)

    MatrixXf X(4,2); 
    VectorXf y(4); 

    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;

    fmt::print("Initializing all classes;\n");
    Dataset data(X,y);

    SearchSpace SS;
    SS.init(data);

    Parameters params;
    params.pop_size = 20; // small pop just for tests
    Population pop = Population<ProgramType::Regressor>(); 

    // aux classes (they are not tested in-depth in this file)
    Evaluation evaluator = Evaluation<ProgramType::Regressor>(params.scorer_);
    Selection selector = Selection<ProgramType::Regressor>(params.sel, false);
    Selection survivor = Selection<ProgramType::Regressor>(params.surv, true);
    Variation variator = Variation<ProgramType::Regressor>(params, SS);
            
    selector.set_operator();
    survivor.set_operator();

    // size, all individuals were initialized
    ASSERT_TRUE(pop.size() == pop.individuals.size()
             && pop.size() == 0); //before initialization, it should be empty

    fmt::print("Initializing individuals in the population:\n");
    pop.init(SS, params);

    fmt::print("pop.size() {}, pop.individuals.size() {}, params.pop_size, {}",
               pop.size(), pop.individuals.size(), params.pop_size);
    ASSERT_TRUE(pop.size() == pop.individuals.size()
             && pop.size()/2 == params.pop_size); // now we have a population.
                                                // Its size is actually the double,
                                                // but the real value goes just up to the middle (no offspring was initialized)

    for (int i=0; i<params.pop_size; ++i)
    {
        fmt::print("{} ", i);
        fmt::print("Individual: {}\n", 
        pop[i].program.get_model("compact", true));
    }

    // print models
    fmt::print("Printing from population method:\n");
    fmt::print("{}\n",pop.print_models()); // may yeld seg fault if string is too large for buffer

    // island sizes increases and comes back to the same values after update
    fmt::print("Performing all steps of an evolution (sequential, not parallel)\n");
    for (int i=0; i<100; ++i) // update and prep offspring slots works properly
    {
        vector<vector<size_t>> survivors(pop.num_islands);

        fmt::print("Fitting individuals\n"); // this must be done in one thread (or implement mutex), because we can have multiple islands pointing to same individuals
        for (int j=0; j<pop.num_islands; ++j)
        {
            fmt::print("Island {}, individuals {}\n", j, pop.get_island_indexes(j));

            // we can calculate the fitness for each island
            fmt::print("Fitness\n");
            evaluator.update_fitness(pop, j, data, params, true, false);
        }

        // TODO: fix random state and make it work with taskflow
        fmt::print("Evolution step {}\n", i);
        for (int j=0; j<pop.num_islands; ++j)
        {
            // just so we can call the update method
            fmt::print("Selection\n");
            vector<size_t> parents = selector.select(pop, j, params);
            ASSERT_TRUE(parents.size() > 0);

            fmt::print("Preparing offspring\n");
            pop.add_offspring_indexes(j);

            // variation applied to population
            fmt::print("Variations for island {}\n", j);
            variator.vary(pop, j, parents);

            fmt::print("fitting {}\n", j); // at this step, we know that theres only one pointer to each individual being fitted, so we can perform it in parallel
            evaluator.update_fitness(pop, j, data, params, true, true);
        
            fmt::print("survivors {}\n", j);
            auto island_survivors = survivor.survive(pop, j, params);
            survivors.at(j) = island_survivors;
        }
        
        fmt::print("Updating and migrating\n");
        pop.update(survivors); 
        fmt::print("Migrating\n");
        pop.migrate();

        fmt::print("Printing generation {} population:\n", i);
        for (int i=0; i<params.pop_size; ++i)
        {
            fmt::print("{} ", i);
            fmt::print("Individual: {}\n", 
            pop[i].program.get_model("compact", true));
        }

        for (int j=0; j<pop.num_islands; ++j)
        {
            fmt::print("Island {}, idxs {}\n", j, pop.get_island_indexes(j));
            for (int k=0; k<pop.get_island_indexes(j).size(); ++k){
                fmt::print("Individual {} (fitness {}): {}\n",
                        pop.get_island_indexes(j).at(k),
                        pop[pop.get_island_indexes(j).at(k)].fitness.values,
                        pop[pop.get_island_indexes(j).at(k)].program.get_model("compact", true));
            }
        }
    }
}

