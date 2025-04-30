#include <gtest/gtest.h>
#include "../../src/eval/evaluation.h"
#include "../../src/eval/metrics.h"
#include "../../src/eval/scorer.h"

using namespace Brush::Eval;

TEST(Evaluation, accuracy)
{
    // test zero one loss
    VectorXf yhat(10), y(10), res(10), loss(10);
	
    y << 0.0,
         1.0,
         0.0,
         0.0,
         1.0,
         0.0,
         0.0,
         1.0,
         0.0,
         1.0;
    
    yhat << 0.0,  // correct
	        1.0,  // correct
	        1.0,  // incorrect
	        0.0,  // correct
	        0.0,  // incorrect
	        1.0,  // incorrect
	        1.0,  // incorrect
	        0.0,  // incorrect
	        0.0,  // correct
	        0.0;  // incorrect
	
    res << 0.0, // should be 40% accuracy
           0.0,
           1.0,
           0.0,
           1.0,
           1.0,
           1.0,
           1.0,
           0.0,
           1.0;
           
    float score = zero_one_loss(y, yhat, loss);
    
    if (loss != res)
    {
        std::cout << "loss:" << loss.transpose() << "\n";
        std::cout << "res:" << res.transpose() << "\n";
    }
    ASSERT_TRUE(loss == res);
    ASSERT_EQ(((int)(score*10000)), 3999);
}


// TEST(EvaluationTest, UpdateFitnessTest) {
//     // TODO: Add test case for update_fitness function
//     Population<ProgramTypeA> population;
//     Dataset data;
//     Parameters params;
//     bool fit = true;
//     bool validation = false;

//     // Add some individuals to the population
//     Individual<ProgramTypeA> ind1;
//     Individual<ProgramTypeA> ind2;
//     population.add_individual(ind1);
//     population.add_individual(ind2);

//     // Call the update_fitness function
//     Evaluation<ProgramTypeA> evaluation;
//     evaluation.update_fitness(population, 0, data, params, fit, validation);

//     // TODO: Add assertions to verify the correctness of the update_fitness function
//     // For example:
//     // ASSERT_EQ(population.get_individual(0).get_fitness(), expected_fitness_0);
//     // ASSERT_EQ(population.get_individual(1).get_fitness(), expected_fitness_1);
// }

// TEST(EvaluationTest, AssignFitTest) {
//     // TODO: Add test case for assign_fit function
// }

// TEST(EvaluationTest, DifferentMetricsTest) {
//     // TODO: Add test case for different metrics
// }

// TEST(EvaluationTest, AnotherMetricTest) {
//     // TODO: Add test case for another metric
// }
