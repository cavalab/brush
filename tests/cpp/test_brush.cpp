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

// TODO: omg i need to figure out why my code only works if i import basically the whole stuff
#include "../../src/selection/selection.cpp"
#include "../../src/selection/selection_operator.cpp"
#include "../../src/selection/nsga2.cpp"
#include "../../src/selection/lexicase.cpp"
#include "../../src/eval/evaluation.cpp"
#include "../../src/pop/archive.cpp"
#include "../../src/pop/population.cpp"

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

    Parameters params;
    params.set_pop_size(100);
    params.set_gens(10);
    params.set_mig_prob(0.0);

     // TODO: archive tests
     // TODO: solve issues from GH

    params.set_verbosity(2); // TODO: verbosity tests

     // checking if validation size works
    params.set_validation_size(0.2);

    std::cout << "n jobs = 1" << std::endl;
    params.set_n_jobs(1);
    Brush::RegressorEngine est5(params);
    est5.run(data); // this will not use validation size from parameters
    std::cout << "best individual using run(data)" << std::endl;
    std::cout << est5.best_ind.program.get_model() << std::endl;
   
    est5.fit(X, y); // this will use validation size from parameters
    std::cout << "best individual using fit(X, y)" << std::endl;
    std::cout << est5.best_ind.program.get_model() << std::endl;
    
    std::cout << "n jobs = 2" << std::endl;
    params.set_n_jobs(2);
    Brush::RegressorEngine est2(params);
    est2.run(data);

    std::cout << "n jobs = -1" << std::endl;
    params.set_n_jobs(-1);
    Brush::RegressorEngine est3(params);
    est3.run(data);

    std::cout << "n jobs = 0" << std::endl;
    params.set_n_jobs(0);
    Brush::RegressorEngine est4(params);
    est4.run(data);

    std::cout << "testing migration" << std::endl;
    
    params.set_pop_size(10);
    params.set_gens(10);
    params.set_mig_prob(0.5);

    std::cout << "n jobs = 1" << std::endl;
    params.set_n_jobs(1);
    Brush::RegressorEngine est6(params);
    est6.run(data);
    
    std::cout << "n jobs = 2" << std::endl;
    params.set_logfile("./tests/cpp/__logfile.csv"); // TODO: test classification and regression and save log so we can inspect it
    params.set_n_jobs(2);
    Brush::RegressorEngine est7(params);
    est7.run(data);
    params.set_logfile("");

    std::cout << "n jobs = -1" << std::endl;
    params.set_n_jobs(-1);
    Brush::RegressorEngine est8(params);
    est8.run(data);

    std::cout << "n jobs = 0" << std::endl;
    params.set_n_jobs(0);
    Brush::RegressorEngine est9(params);
    est9.run(data);

     // when popsize is not divisible by num_islands
    std::cout << "popsize not divisible by num_islands" << std::endl;
    params.set_pop_size(15);
    params.set_gens(10);
    params.set_num_islands(4); // fewer individuals in one island
    params.set_n_jobs(1);
    Brush::RegressorEngine est_not_div1(params);
    est_not_div1.run(data);

    // TODO: logger
    std::cout << "popsize not divisible by num_islands" << std::endl;
    params.set_pop_size(10);
    params.set_gens(10);
    params.set_num_islands(3); // extra individuals in one island
    params.set_n_jobs(1);
    Brush::RegressorEngine est_not_div2(params);
    est_not_div2.run(data);

    // TODO: test predict and predict proba
     // TODO: validation loss
}


TEST(Engine, ClassificationEngineWorks)
{
     // TODO: test classifier and multiclassifier 
    Dataset data = Data::read_csv("docs/examples/datasets/d_analcatdata_aids.csv", "target");
    
    ASSERT_TRUE(data.classification);

    Parameters params;
    params.set_pop_size(100);
    params.set_gens(10);
    params.set_mig_prob(0.0);
    params.set_scorer_("log");

    params.set_verbosity(2); // TODO: verbosity tests

    Brush::ClassifierEngine est(params);
    est.run(data);
}