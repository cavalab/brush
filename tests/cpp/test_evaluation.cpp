#include <gtest/gtest.h>
#include "../../src/eval/evaluation.h"
#include "../../src/eval/metrics.h"
#include "../../src/eval/scorer.h"
#include "../../src/program/functions.h"
#include "../../src/simplification/constants.h"

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

TEST(Evaluation, ScorerRegressionMSE)
{
    VectorXf y(3), yhat(3), loss_expected(3), loss(3);
    y << 1.0, 2.0, 3.0;
    yhat << 1.0, 4.0, 2.0;

    float expected = mse(y, yhat, loss_expected);

    Scorer<PT::Regressor> scorer("mse");
    float actual = scorer.score(y, yhat, loss, {});

    ASSERT_NEAR(actual, expected, 1e-6);
    ASSERT_TRUE(loss.isApprox(loss_expected, 1e-6));
}

TEST(Evaluation, ScorerBinaryAccuracy)
{
    VectorXf y(4), yhat(4), loss_expected(4), loss(4);
    y << 0.0, 1.0, 1.0, 0.0;
    yhat << 0.1, 0.9, 0.2, 0.8;

    float expected = zero_one_loss(y, yhat, loss_expected);

    Scorer<PT::BinaryClassifier> scorer("accuracy");
    float actual = scorer.score(y, yhat, loss, {});

    ASSERT_NEAR(actual, expected, 1e-6);
    ASSERT_TRUE(loss.isApprox(loss_expected, 1e-6));
}

TEST(Evaluation, MulticlassSoftmaxAndMetrics)
{
    ArrayXf first(2), second(2), third(2);
    first  << 2.0f, 0.0f;
    second << 1.0f, 1.0f;
    third  << 0.0f, 2.0f;

    // testing wether softmax return valid probabilities (normalized per row).
    // softmax here is representing 3 classes, and we have 2 rows (2 samples)
    const auto probabilities = Function<NodeType::Softmax>{}(first, second, third);

    ASSERT_EQ(probabilities.rows(), 2);
    ASSERT_EQ(probabilities.cols(), 3);

    ASSERT_NEAR(probabilities.row(0).sum(), 1.0f, 1e-6f);
    ASSERT_NEAR(probabilities.row(1).sum(), 1.0f, 1e-6f);

    EXPECT_GT(probabilities(0, 0), probabilities(0, 1));
    EXPECT_GT(probabilities(1, 2), probabilities(1, 1));

    VectorXf y(2), loss(2);
    y << 0.0f, 2.0f; // y needs to be float, but gets casted to int when checking for hit/miss inside multiclass losses
    // The model has 100% accuracy for this y, so we should expect perfect metric values
    EXPECT_NEAR(mean_multi_log_loss(y, probabilities, loss),
                (-std::log(probabilities(0, 0)) - std::log(probabilities(1, 2))) / 2.0f,
                1e-6f);
    EXPECT_NEAR(multi_zero_one_loss(y, probabilities, loss), 1.0f, 1e-6f);
    EXPECT_NEAR(multi_bal_zero_one_loss(y, probabilities, loss), 1.0f, 1e-6f);
}

TEST(Evaluation, MulticlassSoftmaxHasOneOutputPerClass)
{
    ArrayXXf X(6, 2);
    X << 0.0f, 1.0f,
         1.0f, 0.0f,
         0.5f, 0.5f,
         2.0f, 1.0f,
         1.0f, 2.0f,
         2.0f, 2.0f;
    ArrayXf y(6);
    y << 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f;

    Dataset data(X, y, {}, {}, {}, true);
    SearchSpace search_space(data);
    Parameters params;
    params.classification = true;
    params.set_n_classes(y);
    params.max_depth = 3;
    params.max_size = 20;

    auto program = search_space.make_multiclass_classifier(3, 20, params);

    // root softmax must have one arg (subtree) for each class 
    EXPECT_EQ(program.Tree.begin().node->data.arg_types.size(), params.n_classes);

    // something needs to be tunable. Even without constants, we have the softmax weights
    ASSERT_GT(program.get_n_weights(), 0);

    program.fit(data, {1.0f, 2.0f, 3.0f});

    const auto probabilities = program.predict_proba(data);
    EXPECT_EQ(probabilities.cols(), params.n_classes);

    // each row probability sums up to 1
    for (int row = 0; row < probabilities.rows(); ++row)
        EXPECT_NEAR(probabilities.row(row).sum(), 1.0f, 1e-6f);

    Simpl::Constants_simplifier simplifier;
    simplifier.simplify_tree<PT::MulticlassClassifier>(program, search_space, data);

    // simplification, if applied, does not break softmax
    const auto simplified_probabilities = program.predict_proba(data);
    for (int row = 0; row < simplified_probabilities.rows(); ++row)
        EXPECT_NEAR(simplified_probabilities.row(row).sum(), 1.0f, 1e-6f);
}

TEST(Evaluation, SplitThresholdsAreNotOptimizerParameters)
{
    ArrayXXf X(6, 1);
    X << 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f;
    
    ArrayXf y(6);
    y << 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f;

    Dataset data(X, y, {}, {}, {"ArrayF"});

    RegressorProgram program = json({{"Tree", {
        {{"node_type", "SplitBest"}, {"is_weighted", true}},
        {{"node_type", "Constant"}, {"is_weighted", true}},
        {{"node_type", "Constant"}, {"is_weighted", true}}
    }}, {"is_fitted_", false}});

    program.fit(data);
    const auto prediction = program.predict(data);

    EXPECT_TRUE(prediction.isApprox(y, 1e-3f))
        << "prediction=" << prediction.transpose()
        << ", weights=" << program.get_weights().transpose()
        << ", model=" << program.get_model();

    // The greedy split threshold lives in SplitBest::W and must not enter the
    // Ceres parameter vector; only the two leaf constants do.
    EXPECT_EQ(program.get_weights().size(), 2); // split best should not be considered, so we will have 2 optimizable parameters on this tree
}

TEST(Evaluation, BinaryOptimizerUsesClassWeights)
{
    ArrayXXf X(2, 1);
    X << 0.0f, 1.0f;
    
    ArrayXf y(2);
    y << 0.0f, 1.0f;

    Dataset data(X, y, {}, {}, {}, true);

    // given the X and y above, we should have a perfect classifier here
    ClassifierProgram program = json({{"Tree", {
        {{"node_type", "Logistic"}, {"is_weighted", false}},
        {{"node_type", "OffsetSum"}, {"is_weighted", true}},
        {{"node_type", "Constant"}, {"is_weighted", true}}
    }}, {"is_fitted_", false}});

    program.fit(data, {1.0f, 9.0f});
    
    const auto probability = program.predict_proba(data);

    // one proba will be low, the other will be super high
    EXPECT_GT(probability.mean(), 0.8f);
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
