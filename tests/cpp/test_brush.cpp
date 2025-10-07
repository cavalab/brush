#include "testsHeader.h"

#include "../../src/vary/search_space.h"
#include "../../src/program/program.h"

// #include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"
#include "../../src/engine.h"
#include "../../src/engine.cpp"
#include "../../src/selection/selection.h"
#include "../../src/selection/selection_operator.h"
#include "../../src/selection/nsga2.h"
#include "../../src/selection/lexicase.h"
#include "../../src/eval/evaluation.h"
#include "../../src/pop/archive.h"
#include "../../src/pop/population.h"
#include "../../src/simplification/constants.h"
#include "../../src/simplification/inexact.h"

// TODO: omg i need to figure out why my code only works if i import basically the whole stuff. It seems to be related to templating
#include "../../src/selection/selection.cpp"
#include "../../src/selection/selection_operator.cpp"
#include "../../src/selection/nsga2.cpp"
#include "../../src/selection/lexicase.cpp"
#include "../../src/eval/evaluation.cpp"
#include "../../src/pop/archive.cpp"
#include "../../src/pop/population.cpp"
// #include "../../src/bandit/bandit.cpp"
// #include "../../src/bandit/bandit_operator.cpp"
// #include "../../src/bandit/dummy.cpp"
// #include "../../src/bandit/thompson.cpp"
#include "../../src/simplification/constants.cpp"
#include "../../src/simplification/inexact.cpp"

// TODO: test predict from archive
// TODO: rename it to test_engine 

// TODO: test serialization of archive (get archive and save to json)

// TODO: test logger, verbose, print stats, etc.
TEST(Engine, EngineWorks)
{
    MatrixXf X(10,2);
    ArrayXf y(10);
    X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
         0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

         0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
         0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

    y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
         3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;

    Dataset data(X,y);
    SearchSpace ss(data);

    Parameters params;
    params.set_pop_size(100);
    params.set_max_gens(10);
    params.set_mig_prob(0.0);

     // TODO: archive tests
     
    // TODO: test termination criterion --- max stall, generations, time  

    params.set_verbosity(2); // TODO: verbosity tests

     // checking if validation size works
    params.set_validation_size(0.2);

    std::cout << "n jobs = 1" << std::endl;
    params.set_n_jobs(1);
    Brush::RegressorEngine est5(params, ss);
    est5.run(data); // this will not use validation size from parameters
    std::cout << "best individual using run(data)" << std::endl;
    std::cout << est5.best_ind.program.get_model() << std::endl;
   
    est5.fit(X, y); // this will use validation size from parameters
    std::cout << "best individual using fit(X, y)" << std::endl;
    std::cout << est5.best_ind.program.get_model() << std::endl;
    
    std::cout << "n jobs = 2" << std::endl;
    params.set_n_jobs(2);
    Brush::RegressorEngine est2(params, ss);
    est2.run(data);

    std::cout << "n jobs = -1" << std::endl;
    params.set_n_jobs(-1);
    Brush::RegressorEngine est3(params, ss);
    est3.run(data);

    std::cout << "n jobs = 0" << std::endl;
    params.set_n_jobs(0);
    Brush::RegressorEngine est4(params, ss);
    est4.run(data);

    std::cout << "testing migration" << std::endl;
    
    params.set_mig_prob(0.5);

    // just to see if nothing breaks
    params.set_use_arch(true);

    std::cout << "n jobs = 1" << std::endl;
    params.set_n_jobs(1);
    Brush::RegressorEngine est6(params, ss);
    est6.run(data);
    
    std::cout << "n jobs = 2" << std::endl;
    params.set_logfile("./tests/cpp/__logfile.csv"); // TODO: test classification and regression and save log so we can inspect it
    params.set_n_jobs(2);
    Brush::RegressorEngine est7(params, ss);
    est7.run(data);
    params.set_logfile("");

    std::cout << "n jobs = -1" << std::endl;
    params.set_n_jobs(-1);
    Brush::RegressorEngine est8(params, ss);
    est8.run(data);

    std::cout << "n jobs = 0" << std::endl;
    params.set_n_jobs(0);
    Brush::RegressorEngine est9(params, ss);
    est9.run(data);

     // when popsize is not divisible by num_islands
    std::cout << "popsize not divisible by num_islands" << std::endl;
    params.set_pop_size(15);
    params.set_max_gens(10);
    params.set_num_islands(4); // fewer individuals in one island
    params.set_n_jobs(1);
    Brush::RegressorEngine est_not_div1(params, ss);
    est_not_div1.run(data);

    // TODO: use logger in the tests
    std::cout << "popsize not divisible by num_islands" << std::endl;
    params.set_pop_size(10);
    params.set_max_gens(10);
    params.set_num_islands(3); // extra individuals in one island
    params.set_n_jobs(1);
    Brush::RegressorEngine est_not_div2(params, ss);
    est_not_div2.run(data);

    // TODO: validation loss
}

#include <vector>
#include <string>

class EngineTest : public ::testing::TestWithParam<std::string> {};

TEST_P(EngineTest, ClassificationEngineWorks)
{
    std::string bandit_type = GetParam();
    Dataset data = Data::read_csv("docs/examples/datasets/d_analcatdata_aids.csv", "target");
    
    SearchSpace ss(data);

    ASSERT_TRUE(data.classification);

    Parameters params;
    params.set_pop_size(10);

    // We MUST set these three parameters to run a classification problem
    params.set_n_classes(data.y);
    params.set_class_weights(data.y);
    params.set_sample_weights(data.y);

    params.set_max_gens(1000);
    params.set_bandit(bandit_type);
    params.set_num_islands(1);
    params.set_mig_prob(0.0);

    params.set_verbosity(2);

    // Test with log loss score
    params.set_scorer("log");
    std::cout << "Bandit type: " << bandit_type << std::endl;
    std::cout << "Metric: log" << std::endl;
    Brush::ClassifierEngine est(params, ss);
    est.run(data);

    // Test with average precision score
    params.set_scorer("average_precision_score");
    std::cout << "Bandit type: " << bandit_type << std::endl;
    std::cout << "Metric: average_precision_score" << std::endl;
    Brush::ClassifierEngine est2(params, ss);
    est2.run(data);

    // Test with accuracy score
    params.set_scorer("accuracy");
    std::cout << "Bandit type: " << bandit_type << std::endl;
    std::cout << "Metric: accuracy" << std::endl;
    Brush::ClassifierEngine est3(params, ss);
    est3.run(data);

    // Test with f1 score
    params.set_scorer("balanced_accuracy");
    std::cout << "Bandit type: " << bandit_type << std::endl;
    std::cout << "Metric: balanced_accuracy" << std::endl;
    Brush::ClassifierEngine est4(params, ss);
    est4.run(data);

    std::cout << "Parameters probs:" << std::endl;
    std::cout << "cx: " << est.params.get_cx_prob() << std::endl;
    for (const auto& [name, prob] : est.params.get_mutation_probs())
        std::cout << name << ": " << prob << std::endl;

    std::cout << "Search Space:" << std::endl;
    est.ss.print();
}

INSTANTIATE_TEST_SUITE_P(
    BanditTypes,
    EngineTest,
    ::testing::Values(
        "dummy",
        "thompson",
        "dynamic_thompson"
    )
);

TEST(Engine, SavingLoadingFixedNodes)
{
    // runs a classification problem for 100 generations, save it to a file,
    // then load it, then run for 100 more generations, and checks if all
    // individuals in the population have the logistic node as its root.

    Dataset data = Data::read_csv("docs/examples/datasets/d_analcatdata_aids.csv", "target");

    SearchSpace ss(data);

    Parameters params;
    params.set_verbosity(2);
    params.set_scorer("log");
    params.set_cx_prob(0.0);
    params.set_save_population("./tests/cpp/__pop_clf.json");

    Brush::ClassifierEngine est(params, ss);
    est.run(data);

    cout << "Checking if all individuals in the population have the logistic node as its root after saving the population to a file" << endl;
    for (auto& ind : est.archive.individuals)
    {
        std::cout << "-----" << std::endl << ind.program.get_model() << 
                     ". last variation was: " << ind.variation << endl;

        Node cx_child_root = *(ind.program.Tree.begin());

        cout << "Root has a prob change of: " <<
            std::to_string(cx_child_root.get_prob_change()) << std::endl;

        ASSERT_TRUE(cx_child_root.node_type == NodeType::Logistic);
        ASSERT_TRUE(cx_child_root.get_prob_change()==0.0);
        ASSERT_TRUE(cx_child_root.fixed==true);
    }

    // TODO: why if I set cx_prob to 0.0 it does not work? (maybe because Im using the same params object for the two engines? do i need to remove save_pop file first?)
    
    Parameters params2;
    params2.set_verbosity(2);
    params2.set_scorer("log");
    params2.set_load_population("./tests/cpp/__pop_clf.json");
    
    Brush::ClassifierEngine est2(params2, ss);
    est2.run(data);

    cout << "Checking if all individuals in the population have the logistic node as its root after loading a previously saved pop to resume execution" << endl;
    for (auto& ind : est2.archive.individuals)
    {
        std::cout << "-----" << std::endl << ind.program.get_model() << 
                     ". last variation was: " << ind.variation << endl;

        Node cx_child_root = *(ind.program.Tree.begin());

        cout << "Root has a prob change of: " <<
            std::to_string(cx_child_root.get_prob_change()) << std::endl;

        ASSERT_TRUE(cx_child_root.node_type == NodeType::Logistic);
        ASSERT_TRUE(cx_child_root.get_prob_change()==0.0);
        ASSERT_TRUE(cx_child_root.fixed==true);
    }
}


TEST(Engine, MaxStall)
{
    MatrixXf X(10,2);
    ArrayXf y(10);
    X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
         0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

         0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
         0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

    y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
         3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;

    Dataset data(X,y);
    SearchSpace ss(data);

    Parameters params;
    params.set_pop_size(100);
    params.set_max_gens(10000000);
    params.set_mig_prob(0.0);
    params.set_max_stall(10);
    params.set_verbosity(1); // change here for more information. Im keeping it short to avoid long logs

    cout << "Testing max stall termination criterion" << endl;
    cout << "using 10B generations. The test should not freeze." << endl;
    Brush::RegressorEngine est(params, ss);
    est.run(data);
}


// test working with population files
// create a population for 2 generations, save it, then load it again. Do
// it several times so we can test different initializations
TEST(Engine, engine_save_load_pop_works)
{
    // This dataset was causing RuntimeError: [json.exception.type_error.302] type must be number, but is null.
    // I decided to use it for testing then
    Dataset data = Data::read_csv("./docs/examples/datasets/d_analcatdata_aids.csv", "target");

    SearchSpace ss(data);

    std::unordered_map<string, float> f = {
        {"SplitBest", 1.0},
        {"Add", 1.0},
        {"Mul", 1.0},
        {"Sin", 1.0},
        {"Cos", 1.0},
        {"Exp", 1.0},
        {"Logabs", 1.0}
    };

    Parameters params_save;
    params_save.set_functions(f);
    params_save.set_pop_size(200);
    params_save.set_max_gens(10);
    params_save.set_scorer("log");
    params_save.set_verbosity(1);
    params_save.set_use_arch(false);
    params_save.set_save_population("./tests/cpp/__pop_analcatdata_aids.json");

    Parameters params_load;
    params_load.set_functions(f);
    params_load.set_pop_size(200);
    params_load.set_max_gens(10);
    params_load.set_scorer("average_precision_score");
    params_load.set_verbosity(1);
    params_load.set_use_arch(true);
    params_load.set_load_population("./tests/cpp/__pop_analcatdata_aids.json");

    for (int run = 0; run < 10; ++run) {
        Brush::ClassifierEngine est_save(params_save, ss);
        est_save.fit(data);
        
        Brush::ClassifierEngine est_load(params_load, ss);
        est_load.fit(data);
    }
}

// brute forcing errors with d_enc dataset. This is extremely slow, and I use
// it to test very rare events.
TEST(Engine, DEnc)
{
    // Dataset data = Data::read_csv("./docs/examples/datasets/d_enc.csv", "label");
    Dataset data = Data::read_csv("./docs/examples/datasets/d_example_patients.csv", "target");

    SearchSpace ss(data);

    std::vector<std::string> bandits = {"dummy", "thompson", "dynamic_thompson"};

    for (const auto& bandit : bandits) {
        std::cout << "Running bandit: " << bandit << std::endl;
        for (int run = 0; run < 2; ++run) {
            Parameters params;
            params.set_pop_size(100);
            params.set_max_gens(50);
            params.set_max_stall(100); // avoid early stopping
            params.set_max_depth(10);
            params.set_objectives({"scorer", "linear_complexity"});
            params.set_bandit(bandit);
            params.set_weights_init(false);
            params.set_use_arch(false);
            
            // params.set_constants_simplification(false);
            // params.set_inexact_simplification(false);

            params.set_verbosity(1);

            Brush::RegressorEngine est(params, ss);
            est.fit(data);

            std::cout << "model: " << est.best_ind.program.get_model() << std::endl;
        }
    }
}
