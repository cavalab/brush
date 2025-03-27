#include "testsHeader.h"
#include "../../src/vary/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"

using testing::TestWithParam;

// Hashes corresponding to a 3-ary Prod operator
const std::size_t sig_hash      = 5617655905677279916u;
const std::size_t sig_dual_hash = 10188582206427064428u;
const std::size_t complete_hash = 1786662244046809282u;

class OptimizerTest 
    : public TestWithParam< std::tuple<string,json,std::function<bool(ArrayXf)>> > {
    /** @brief Texture used to create value-parameterized tests. Expects a tuple
     * with <string dataset_name, json PRGjson, Lambda check_fit>. These values
     * are used to create a program based on `PRGjson`, fit it with the given 
     * dataset with filename `dataset_name`, and check if the adjusted weights
     * are as expected by the `check_fit` function.
    */
    protected:
        void SetUp() override {
            // Unpack test settings into the variables
            std::tie(dataset_name, PRGjson, check_fit) = GetParam();
        }
        // void TearDown() override { }

        // Those parameters will be accessible inside TEST_P(OptimizerTest, ...)
        string dataset_name;
        json PRGjson;
        std::function<bool(ArrayXf)> check_fit;
};

TEST_P(OptimizerTest, OptimizeWeightsWorksCorrectly) {
    /* @brief Tests whether weight optimization works on simple problems.
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */

    Dataset data = Data::read_csv(dataset_name,"target");

    SearchSpace SS;
    SS.init(data);

    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);

    // predict values
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);
    fmt::print( "y_pred: {}\n", y_pred);

    auto learned_weights = PRG.get_weights();
    fmt::print( "weights: {}\n", learned_weights);

    // calculating the MSE
    float mse_error = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-3)) << "Not all predictions " 
        "are close to the correct values. Predictions are\n" << y_pred <<
        "\nwhile correct values are\n" << data.y << std::endl;
    
    ASSERT_TRUE(check_fit(learned_weights)) << "Check of learned weights "
        "didn't pass. Learned weights are\n" << learned_weights << std::endl;

    ASSERT_TRUE(mse_error <= 1e-3) << "The MSE " << mse_error << "obtained after fitting "
        "the expression is not smaller than threshold of 1e-3" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(OptimizerTestParameters, OptimizerTest,
    testing::Values(
        /** Simple additive problem with positive weights.
         *  The dataset models y = 2*x1 + 3*x2. 
         *  The initial model is yhat = 1*x1 + 1*x2. 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_plus_3x2.csv",
            json({{"Tree", {
                { {"node_type","Add"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", true } }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(2);
                true_weights << 2.0, 3.0;
                return learned_weights.isApprox(true_weights, 1e-3);
            }
        ),

        /** Simple additive problem with one negative weight.
         *  The dataset models y = 2*x1 - 3*x2. 
         *  The initial model is yhat = 1*x1 + 1*x2. 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_subtract_3x2.csv",
            json({{"Tree", {
                { {"node_type","Add"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", true } }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(2);
                true_weights << 2.0, -3.0;
                return learned_weights.isApprox(true_weights, 1e-3);
            }
        ),

        /** Simple subtractive problem with positive weights.
         *  The dataset models y = 2*x1 - 3*x2. 
         *  The initial model is yhat = 1*x1 - 1*x2. 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_subtract_3x2.csv",
            json({{"Tree", {
                { {"node_type","Sub"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", true } }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(2);
                true_weights << 2.0, 3.0;
                return learned_weights.isApprox(true_weights, 1e-2);
            }
        ),

        // TODO: make examples hardcoded here and get rid of lots of data? or even have one single example and calculate the targets on the fly
        /** Simple subtractive problem with one negative weight.
         *  The dataset models y = 2*x1 + 3*x2. 
         *  The initial model is yhat = 1*x1 - 1*x2. 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_plus_3x2.csv",
            json({{"Tree", {
                { {"node_type","Sub"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", true } }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(2);
                true_weights << 2.0, -3.0;
                return learned_weights.isApprox(true_weights, 1e-3);
            }
        ),

        /** Simple multiplication problem with weighted terminal.
         *  The dataset models y = 2*x1 * 3*x2. 
         *  The initial model is yhat = (1*x1) * x2. 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_multiply_3x2.csv",
            json({{"Tree", {
                { {"node_type","Mul"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool {
                return abs(6.0 - learned_weights.prod()) <= 1e-3;
            }
        ),

        /** Simple multiplication problem with weighted Mul op.
         *  The dataset models y = 2*x1 * 3*x2. 
         *  The initial model is yhat = 1*(x1 * x2). 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_multiply_3x2.csv",
            json({{"Tree", {
                { {"node_type","Mul"     },                   {"is_weighted", true} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                return abs(6.0 - learned_weights(0)) <= 1e-3;
            }
        ),

        /** Simple division problem with weighted terminal.
         *  The dataset models y = 2*x1 / 3*x2. 
         *  The initial model is yhat = (1*x1) / x2. 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_divide_3x2.csv",
            json({{"Tree", {
                { {"node_type","Div"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                return abs(2.0/3.0 - learned_weights(0)) <= 1e-3;
            }
        ),

        /** Simple division problem with weighted Div op.
         *  The dataset models y = 2*x1 / 3*x2. 
         *  The initial model is yhat = 1*(x1 / x2). 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_divide_3x2.csv",
            json({{"Tree", {
                { {"node_type","Div"     },                   {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                return abs(2.0/3.0 - learned_weights(0)) <= 1e-3;
            }
        ),

        /** Simple single variable weighted transformation problem.
         *  The dataset models y = 2*sqrt(x1).
         *  The initial model is yhat = 1*sqrt(x1).
         */
        std::make_tuple (
            "docs/examples/datasets/d_2_sqrt_x1.csv",
            json({{"Tree", {
                { {"node_type","Sqrt"    },                   {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                // 2*sqrt(x1) is equivalent to sqrt(4*x1). The weight of a function
                // node is applied to each of its children, not to the node itself.
                ArrayXf true_weights(1);
                true_weights << 2.0;
                return learned_weights.isApprox(true_weights, 1e-3);            
            }
        ),

        /** Simple single weighted variable transformation problem.
         *  The dataset models y = 2*sqrt(x1).
         *  The initial model is yhat = sqrt(1*x1).
         */
        std::make_tuple (
            "docs/examples/datasets/d_2_sqrt_x1.csv",
            json({{"Tree", {
                { {"node_type","Sqrt"    },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                // 2*sqrt(x1) is equivalent to sqrt(4*x1). Here the weight of
                // is applied to sqrt's child, not to the operator.
                ArrayXf true_weights(1);
                true_weights << 4.0;
                return learned_weights.isApprox(true_weights, 1e-3);            
            }
        ),

        /** Simple single variable weighted transformation problem.
         *  The dataset models y = 2*sin(x1).
         *  The initial model is yhat = 1*sin(x1).
         */
        std::make_tuple (
            "docs/examples/datasets/d_2_sin_x1.csv",
            json({{"Tree", {
                { {"node_type","Sin"     },                   {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(1);
                true_weights << 2.0;
                return learned_weights.isApprox(true_weights, 1e-3);            
            }
        ),

        /** Simple single weighted variable transformation problem.
         *  The dataset models y = sin(0.25*x1).
         *  The initial model is yhat = sin(1*x1).
         */
        std::make_tuple (
            "docs/examples/datasets/d_sin_0_25x1.csv",
            json({{"Tree", {
                { {"node_type","Sin"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(1);
                true_weights << 0.25;
                return learned_weights.isApprox(true_weights, 1e-3);            
            }
        ),

        /** Notable product problem with two variables.
         *  The dataset models y = square(x1) + 2*x1*x2 + square(x2). 
         *  The initial model is yhat = square(1*x1) + 1*x1*x2 + square(x2). 
         */
        std::make_tuple (
            "docs/examples/datasets/d_square_x1_plus_2_x1_x2_plus_square_x2.csv",
            json({{"Tree", {
                { {"node_type","Add"     },                   {"is_weighted", false} },
                { {"node_type","Mul"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} },
                { {"node_type","Add"     },                   {"is_weighted", false} },
                { {"node_type","Square"  },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Square"  },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(2);
                true_weights << 2.0, 1.0;
                return learned_weights.isApprox(true_weights, 1e-3);            
            }
        ),

        /** Product problem with 3-ary Prod operator with one weighted variable.
         *  The dataset models y =  5*x1*x2*x3. 
         *  The initial model is yhat = Prod(1*x1, x2, x3).
         */
        std::make_tuple (
            "docs/examples/datasets/d_5x1_multiply_x2_multiply_x3.csv",
            json({{"Tree", {
                {   // Creating a 3-ary Prod node
                    {"is_weighted",false}, {"node_type","Prod"},

                    // We need to provide this information to avoid letting the parser
                    // infer the node signature
                    {"arg_types"    ,{"ArrayF", "ArrayF", "ArrayF"}},
                    {"ret_type"     ,"ArrayF"},
                    {"sig_hash"     ,sig_hash},
                    {"sig_dual_hash",sig_dual_hash},
                    {"complete_hash",complete_hash} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", true } },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x3"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(1);
                true_weights << 5.0;
                return learned_weights.isApprox(true_weights, 1e-3);            
            }
        ),

        /** Product problem with weighted 3-ary Prod operator.
         *  The dataset models y =  5*x1*x2*x3. 
         *  The initial model is yhat = 1*Prod(x1, x2, x3).
         */
        std::make_tuple (
            "docs/examples/datasets/d_5x1_multiply_x2_multiply_x3.csv",
            json({{"Tree", {
                {   // Creating a 3-ary Prod node
                    {"is_weighted",true}, {"node_type","Prod"},

                    // We need to provide this information to avoid letting the parser
                    // infer the node signature
                    {"arg_types"    ,{"ArrayF", "ArrayF", "ArrayF"}},
                    {"ret_type"     ,"ArrayF"},
                    {"sig_hash"     ,sig_hash},
                    {"sig_dual_hash",sig_dual_hash},
                    {"complete_hash",complete_hash} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x3"}, {"is_weighted", false} }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(1);
                true_weights << 5.0;
                return learned_weights.isApprox(true_weights, 1e-3);            
            }
        ),

        /** Simple weighted constant problem.
         * The dataset models y = 2*x1 * 3*x2. 
         * The initial model is yhat = x1 * x2 * (1*C). 
         */
        std::make_tuple (
            "docs/examples/datasets/d_2x1_multiply_3x2.csv",
            json({{"Tree", {
                { {"node_type","Mul"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x1"}, {"is_weighted", false} },
                { {"node_type","Mul"     },                   {"is_weighted", false} },
                { {"node_type","Terminal"}, {"feature","x2"}, {"is_weighted", false} },
                { {"node_type","Constant"}, {"feature","C" }, {"is_weighted", true } }
            }}, {"is_fitted_",false}}),
            [](ArrayXf learned_weights) -> bool { 
                ArrayXf true_weights(1);
                true_weights << 6.0;
                return learned_weights.isApprox(true_weights, 1e-3);            
            }
        )
    )                        
);