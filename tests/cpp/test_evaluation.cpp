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

// Expected values in the tests below were computed with sklearn.metrics
// (precision_score, recall_score, roc_auc_score, average_precision_score).
class BinaryMetrics : public ::testing::Test {
protected:
    VectorXf y, proba, misclassified;

    void SetUp() override {
        y.resize(10); proba.resize(10); misclassified.resize(10);

        y     << 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,  1.0;
        proba << 0.1, 0.9, 0.4, 0.6, 0.8, 0.3, 0.7, 0.2, 0.05, 0.55;

        // threshold 0.5 -> TP = 3 (idx 1, 4, 9), FP = 2 (idx 3, 6), FN = 2 (idx 2, 7)
        misclassified << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;
    }
};

TEST_F(BinaryMetrics, Precision)
{
    VectorXf loss;
    EXPECT_NEAR(precision_score(y, proba, loss), 0.6f, 1e-6f);  // 3 / (3 + 2)
    ASSERT_TRUE(loss == misclassified);

    // class weights act as sample weights: 3*2 / (3*2 + 2*1)
    EXPECT_NEAR(precision_score(y, proba, loss, {1.0f, 2.0f}), 0.75f, 1e-6f);
    // the loss vector is not weighted (it is used by lexicase)
    ASSERT_TRUE(loss == misclassified);
}

TEST_F(BinaryMetrics, Recall)
{
    VectorXf loss;
    EXPECT_NEAR(recall_score(y, proba, loss), 0.6f, 1e-6f);  // 3 / (3 + 2)
    ASSERT_TRUE(loss == misclassified);

    // recall only looks at positives, so class weights cancel out
    EXPECT_NEAR(recall_score(y, proba, loss, {1.0f, 2.0f}), 0.6f, 1e-6f);
}

TEST_F(BinaryMetrics, PrecisionRecallWithoutPredictedPositives)
{
    // nothing is predicted as positive: zero_division=0, like sklearn
    VectorXf loss;
    VectorXf low = VectorXf::Constant(10, 0.1f);
    EXPECT_NEAR(precision_score(y, low, loss), 0.0f, 1e-6f);
    EXPECT_NEAR(recall_score(y, low, loss), 0.0f, 1e-6f);
    ASSERT_TRUE(loss == y); // every positive is a miss
}

TEST_F(BinaryMetrics, RocAuc)
{
    VectorXf loss;
    EXPECT_NEAR(roc_auc_score(y, proba, loss), 0.72f, 1e-6f);

    // per-sample loss is the log loss
    VectorXf expected_loss = log_loss(y, proba);
    ASSERT_TRUE(loss.isApprox(expected_loss, 1e-6f));

    // AUROC is invariant to scaling all weights of one class
    EXPECT_NEAR(roc_auc_score(y, proba, loss, {1.0f, 2.0f}), 0.72f, 1e-6f);
}

TEST_F(BinaryMetrics, RocAucTiedScores)
{
    // tied scores are a single threshold (trapezoid, not a staircase)
    VectorXf loss, tied(10);
    tied << 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.2, 0.5;
    EXPECT_NEAR(roc_auc_score(y, tied, loss), 0.78f, 1e-6f);

    // constant predictions carry no ranking information
    VectorXf constant = VectorXf::Constant(10, 0.5f);
    EXPECT_NEAR(roc_auc_score(y, constant, loss), 0.5f, 1e-6f);
}

TEST_F(BinaryMetrics, RocAucEdgeCases)
{
    VectorXf loss, perfect(10), reversed(10);
    perfect  = y * 0.8f + VectorXf::Constant(10, 0.1f);
    reversed = VectorXf::Constant(10, 1.0f) - perfect;

    EXPECT_NEAR(roc_auc_score(y, perfect, loss), 1.0f, 1e-6f);
    EXPECT_NEAR(roc_auc_score(y, reversed, loss), 0.0f, 1e-6f);

    // undefined with a single class: we return 0.5 instead of throwing
    VectorXf ones = VectorXf::Ones(10);
    EXPECT_NEAR(roc_auc_score(ones, proba, loss), 0.5f, 1e-6f);
}

TEST(Evaluation, ScorerBinaryNewMetrics)
{
    VectorXf y(4), yhat(4), loss_expected, loss;
    y << 0.0, 1.0, 1.0, 0.0;
    yhat << 0.1, 0.9, 0.2, 0.8;

    Scorer<PT::BinaryClassifier> scorer("precision");
    ASSERT_NEAR(scorer.score(y, yhat, loss, {}), precision_score(y, yhat, loss_expected), 1e-6);
    ASSERT_TRUE(loss.isApprox(loss_expected, 1e-6));

    scorer.set_scorer("recall");
    ASSERT_NEAR(scorer.score(y, yhat, loss, {}), recall_score(y, yhat, loss_expected), 1e-6);
    ASSERT_TRUE(loss.isApprox(loss_expected, 1e-6));

    scorer.set_scorer("roc_auc");
    ASSERT_NEAR(scorer.score(y, yhat, loss, {}), roc_auc_score(y, yhat, loss_expected), 1e-6);
    ASSERT_TRUE(loss.isApprox(loss_expected, 1e-6));
}

class MulticlassMetrics : public ::testing::Test {
protected:
    VectorXf y;
    ArrayXXf proba;

    void SetUp() override {
        y.resize(6); proba.resize(6, 3);

        y << 0.0, 1.0, 2.0, 0.0, 1.0, 2.0;
        proba << 0.7, 0.2,  0.1,
                 0.3, 0.4,  0.3,
                 0.2, 0.5,  0.3,   // predicts 1, true 2
                 0.4, 0.35, 0.25,
                 0.1, 0.3,  0.6,   // predicts 2, true 1
                 0.1, 0.1,  0.8;
    }
};

TEST_F(MulticlassMetrics, PrecisionRecall)
{
    VectorXf loss, misclassified(6);
    misclassified << 0.0, 0.0, 1.0, 0.0, 1.0, 0.0;

    // per class precision = recall = {1, 0.5, 0.5}
    EXPECT_NEAR(multi_precision_score(y, proba, loss), 2.0f/3.0f, 1e-6f);
    ASSERT_TRUE(loss == misclassified);

    EXPECT_NEAR(multi_recall_score(y, proba, loss), 2.0f/3.0f, 1e-6f);
    ASSERT_TRUE(loss == misclassified);
}

TEST_F(MulticlassMetrics, RocAucAndAveragePrecision)
{
    VectorXf loss;
    VectorXf expected_loss = multi_log_loss(y, proba);

    EXPECT_NEAR(multi_roc_auc_score(y, proba, loss), 0.8125f, 1e-6f);
    ASSERT_TRUE(loss.isApprox(expected_loss, 1e-6f));

    // weights change the one-vs-rest problems, since "rest" mixes classes
    EXPECT_NEAR(multi_roc_auc_score(y, proba, loss, {1.0f, 2.0f, 3.0f}),
                0.7708333f, 1e-5f);

    EXPECT_NEAR(multi_average_precision_score(y, proba, loss), 0.75f, 1e-6f);
    ASSERT_TRUE(loss.isApprox(expected_loss, 1e-6f));
}

TEST_F(MulticlassMetrics, AbsentClass)
{
    VectorXf loss, y_absent(6);
    y_absent << 0.0, 1.0, 1.0, 0.0, 1.0, 0.0; // class 2 never occurs

    // ranking metrics skip classes that are absent from y
    EXPECT_NEAR(multi_roc_auc_score(y_absent, proba, loss), 0.8055556f, 1e-5f);
    EXPECT_NEAR(multi_average_precision_score(y_absent, proba, loss), 0.875f, 1e-5f);

    // precision/recall average over classes in y or in the predictions (class
    // 2 is predicted once, so it counts with precision = recall = 0)
    EXPECT_NEAR(multi_precision_score(y_absent, proba, loss), 2.0f/3.0f, 1e-6f);
    EXPECT_NEAR(multi_recall_score(y_absent, proba, loss), 4.0f/9.0f, 1e-6f);
}

TEST_F(MulticlassMetrics, PerfectPredictions)
{
    VectorXf loss;
    ArrayXXf perfect = ArrayXXf::Constant(6, 3, 0.1f);
    for (int i = 0; i < y.size(); ++i)
        perfect(i, static_cast<int>(y(i))) = 0.8f;

    EXPECT_NEAR(multi_precision_score(y, perfect, loss), 1.0f, 1e-6f);
    EXPECT_NEAR(multi_recall_score(y, perfect, loss), 1.0f, 1e-6f);
    EXPECT_NEAR(multi_roc_auc_score(y, perfect, loss), 1.0f, 1e-6f);
    EXPECT_NEAR(multi_average_precision_score(y, perfect, loss), 1.0f, 1e-6f);
}

TEST(Evaluation, ScorerMulticlassNewMetrics)
{
    VectorXf y(3), loss_expected, loss;
    y << 0.0, 1.0, 2.0;
    ArrayXXf proba(3, 3);
    proba << 0.6, 0.3, 0.1,
             0.5, 0.3, 0.2,
             0.1, 0.2, 0.7;

    Scorer<PT::MulticlassClassifier> scorer("precision");
    ASSERT_NEAR(scorer.score(y, proba, loss, {}), multi_precision_score(y, proba, loss_expected), 1e-6);
    ASSERT_TRUE(loss.isApprox(loss_expected, 1e-6));

    scorer.set_scorer("recall");
    ASSERT_NEAR(scorer.score(y, proba, loss, {}), multi_recall_score(y, proba, loss_expected), 1e-6);

    scorer.set_scorer("roc_auc");
    ASSERT_NEAR(scorer.score(y, proba, loss, {}), multi_roc_auc_score(y, proba, loss_expected), 1e-6);
    ASSERT_TRUE(loss.isApprox(loss_expected, 1e-6));

    scorer.set_scorer("average_precision_score");
    ASSERT_NEAR(scorer.score(y, proba, loss, {}), multi_average_precision_score(y, proba, loss_expected), 1e-6);
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
    params.set_random_state(42);
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
