#include "testsHeader.h"
#include "../../src/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"

TEST(Program, OptimizeAdditionPositiveWeights)
{
    /* @brief Tests whether weight optimization works on a simple additive problem.
        The dataset models y = 2*x1 + 3*x2. 
        The initial model is yhat = 1*x1 + 1*x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_plus_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Add"},
            {"is_weighted", false}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
            {"is_weighted", true}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);
    fmt::print( "y_pred: {}\n", y_pred);

    auto learned_weights = PRG.get_weights();
    fmt::print( "weights: {}\n", learned_weights);
    ArrayXf true_weights(2);
    true_weights << 2.0, 3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeAdditionNegativeWeights)
{
    /* @brief Tests whether weight optimization works on a simple additive problem.
        The dataset models y = 2*x1 - 3*x2. 
        The initial model is yhat = 1*x1 + 1*x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_subtract_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Add"},
            {"is_weighted", false}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
            {"is_weighted", true}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();
    ArrayXf true_weights(2);
    true_weights << 2.0, -3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
}

TEST(Program, OptimizeSubtractionPositiveWeights)
{
    /* @brief Tests whether weight optimization works on a simple subtractive problem.
        The dataset models y = 2*x1 - 3*x2. 
        The initial model is yhat = 1*x1 - 1*x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_subtract_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Sub"},
            {"is_weighted", false}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
            {"is_weighted", true}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();
    ArrayXf true_weights(2);
    true_weights << 2.0, 3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeSubtractionNegativeWeights)
{
    /* @brief Tests whether weight optimization works on a simple subtractive problem.
        The dataset models y = 2*x1 + 3*x2. 
        The initial model is yhat = 1*x1 - 1*x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_plus_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Sub"},
            {"is_weighted", false}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
            {"is_weighted", true}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();
    ArrayXf true_weights(2);
    true_weights << 2.0, -3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeMultiply)
{
    /* @brief Tests whether weight optimization works on a simple multiplicative problem.
        The dataset models y = 2*x1 * 3*x2. 
        The initial model is yhat = 1*x1 * x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_multiply_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Mul"},
            {"is_weighted", false}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
            {"is_weighted", false}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    // not necessarily the weights will be exactly equal to the original expression
    // (since it has a non-unique solution). We need to check if the product of 
    // the fitted weights a' anb b' are equal to the product of the expected weights.
    auto learned_weights = PRG.get_weights();
    ArrayXf true_weights(2);
    true_weights << 2.0, 3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(abs(true_weights.prod() - learned_weights.prod()) <= 1e-4);
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeDivide)
{
    /* @brief Tests whether weight optimization works on a simple divisible problem.
        The dataset models y = 2*x1 / 3*x2. 
        The initial model is yhat = 1*x1 / x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_divide_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Div"},
            {"is_weighted", false}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
            {"is_weighted", false}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    // not necessarily the weights will be exactly equal to the original expression
    // (since it has a non-unique solution). We need to check if the ratio of 
    // the fitted weights a' anb b' are equal to the ratio of the expected weights.
    auto learned_weights = PRG.get_weights();

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(abs(2.0/3.0 - learned_weights(0)) <= 1e-3);
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeSqrtOuterWeight)
{
    /* @brief Tests whether weight optimization works on a single variable problem
        with the application of a transformation function.
        The dataset models y = 2*sqrt(x1). 
        The initial model is yhat = 1*sqrt(x1). 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2_sqrt_x1.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Sqrt"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", false}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();

    // 2*sqrt(x1) is equivalent to sqrt(4*x1). The weight of a function
    // node is applied to each of its children, not to the node itself.
    ArrayXf true_weights(1);
    true_weights << 2.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeSqrtInnerWeight)
{
    /* @brief Tests whether weight optimization works on a single variable problem
        with the application of a transformation function.
        The dataset models y = 2*sqrt(x1). 
        The initial model is yhat = sqrt(1*x1). 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2_sqrt_x1.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Sqrt"},
            {"is_weighted", false}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", true}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();

    // 2*sqrt(x1) is equivalent to sqrt(4*x1)
    ArrayXf true_weights(1);
    true_weights << 4.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeSinOuterWeight)
{
    /* @brief Tests whether weight optimization works on a single variable problem
        with the application of a transformation function.
        The dataset models y = 2*sin(x1). 
        The initial model is yhat = 1*sin(x1). 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2_sin_x1.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Sin"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", false}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();

    ArrayXf true_weights(1);
    true_weights << 2.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 


TEST(Program, OptimizeSinInnerWeight)
{
    /* @brief Tests whether weight optimization works on a single variable problem
        with the application of a transformation function.
        The dataset models y = sin(0.25*x1). 
        The initial model is yhat = sin(1*x1). 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_sin_0_25x1.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Sin"},
            {"is_weighted", false}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
            {"is_weighted", true}
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();

    std::cout << learned_weights << std::endl;

    ArrayXf true_weights(1);
    true_weights << 0.25;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-2));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-2));
    ASSERT_TRUE(mse <= 1e-2);
} 