#include "testsHeader.h"

#include "../../src/individual.cpp"
#include "../../src/population.cpp" // TODO: figure out if thats ok to include cpps instead of headers
#include "../../src/eval/evaluation.cpp"
#include "../../src/selection/nsga2.cpp"
#include "../../src/selection/selection.cpp"

using namespace Brush::Pop;
using namespace Brush::Sel;
using namespace Brush::Eval;

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
             && pop.size() == params.pop_size);

    fmt::print("Initializing individuals in the population:\n");
    pop.init(SS, params);
    for (auto& ind : pop.individuals)
    {
        fmt::print("Individual: {}\n", ind->program.get_model("compact", true));
    }

    // print models
    fmt::print("Printing from population method:\n{}\n", pop.print_models());

    // no overlap in island indexes

    fmt::print("Testing island ranges\n");
    // for (std::size_t i = 0; i < pop.island_ranges.size() - 1; ++i) {
    //     int last = std::get<1>(pop.island_ranges.at(i));
    //     int next_first = std::get<0>(pop.island_ranges.at(i+1));

    //     //(last index from one island is EQUAL than first) (no gaps between island)
    //     // (this assumes that we will never iterate to the last index in for loops. TODO: make sure we dont)
    //     ASSERT_TRUE(last == next_first);

    //     // difference between island sizes is at most 1
    //     auto delta = last - std::get<0>(pop.island_ranges.at(i));
    //     auto next_delta = std::get<1>(pop.island_ranges.at(i+1)) - next_first;
    //     ASSERT_TRUE(delta <= next_delta+1 && next_delta <= delta+1);
    // }

    // island sizes increases and comes back to the same values after update
    // fmt::print("Performing all steps of an evolution\n");
    // auto original_islands = pop.island_ranges;
    // for (int i=0; i<10; ++i) // update and prep offspring slots works properly
    // {   // wax on wax off
        
    //     fmt::print("Evaluating population\n");
    //     vector<vector<size_t>> survivors(pop.n_islands);
    //     vector<vector<size_t>> island_parents(pop.n_islands);
        
    //     for (int j=0; j<pop.n_islands; ++j)
    //     {
    //         fmt::print("Island {}, range [{}, {}]\n", j,
    //         std::get<0>(pop.get_island_range(j)), 
    //         std::get<1>(pop.get_island_range(j)) );

    //         fmt::print("Fitness\n");
    //         // we can calculate the fitness for each island
    //         evaluator.fitness(pop, pop.get_island_range(j), data, params, true, false);

    //         fmt::print("Selection\n");
    //         // just so we can call the update method
    //         vector<size_t> parents = selector.select(pop, pop.get_island_range(j), params, data);
            
    //         ASSERT_TRUE(parents.size() > 0);
    //         fmt::print("Updating parents\n");
    //         island_parents.at(j) = parents;

    //         fmt::print("Preparing offspring\n");
    //         pop.prep_offspring_slots(j);

    //         fmt::print("Variations for island {}\n", j);
    //         // variation applied to population
    //         variator.vary(pop, pop.get_island_range(j), island_parents.at(j));

    //         fmt::print("fitting {}\n", j);
    //         evaluator.fitness(pop, pop.get_island_range(j), data, params, true, true);
        
    //         fmt::print("survivors\n", j);
    //         auto island_survivors = survivor.survive(pop, pop.get_island_range(j), params, data);

    //         survivors.at(j) = island_survivors;
    //     }
        
    //     fmt::print("Updating and migrating\n");

    //     // TODO: UPDATE SHOULD SORT SURVIVOR LIST AND REMOVE REPEATED VALUES
    //     pop.update(survivors); 
    //     ASSERT_TRUE(pop.size() == params.pop_size);

    //     pop.migrate();
    //     ASSERT_TRUE(pop.size() == params.pop_size);

    //     fmt::print("Printing generation {} population:\n{}\n", i, pop.print_models());
    // }
    // ASSERT_TRUE(original_islands == pop.island_ranges);
}

