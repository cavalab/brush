#include "testsHeader.h"
#include "../../src/vary/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"

TEST(SearchSpace, Initialization)
{
    float minimum_prob = 1e-1f; // minimum probability of changing
    
    ArrayXf y(4); 
    y << 3.00000,  3.59876, 7.18622, 15.19294;

    // variables have different pairwise correlations with y. The idea is to 
    // see the mutation weights for each floating variable. The slope were
    // calculated using python np.cov(xprime, yprime)[0][1]/np.var(xprime),
    // where xprime and yprime are the z-score normalized arrays obtained
    // from x and y.
    MatrixXf X(5,4); 
    X <<       0,       0,       1,       1, // x0, binary, expected weight=1
               2,       0,       1,       2, // x1, categorical, expected weight=1
         0.05699, 0.62737, 0.72406, 0.99294, // x2, slope ~= 1.069
         0.03993, 0.36558, 0.01393, 0.25878, // x3, slope ~= 0.25
         5.17539, 7.63579,-2.82560, 0.24645; // x4, slope ~= -0.799
    X.transposeInPlace(); // 4 rows x 5 variables

	Dataset dt(X, y);

    // different weights to check if searchspace is initialized correctnly
    unordered_map<string, float> user_ops = {
        {"Add",   1},
        {"Sub",   1},
        {"Div",  .5},
        {"Mul", 0.5}
    };

    SearchSpace SS;
    SS.init(dt, user_ops);

    dt.print();
    SS.print();
    // dtable_fit.print();
    // dtable_predict.print();

    // manually calculated. last value is the avg of prev values
    ArrayXf expected_weights_Xf(4); // 5 elements (x3, x4, x5, c, meanLabel)    
    expected_weights_Xf << 0.80240685, 0.19270448, 0.5994426, 0.531518, 0.531518;
    
    auto actual_weights_f = SS.terminal_weights.at(DataType::ArrayF);
    Eigen::Map<ArrayXf> actual_weights_Xf(actual_weights_f.data(), actual_weights_f.size());
    
    ASSERT_TRUE(expected_weights_Xf.isApprox(actual_weights_Xf));
    
    ArrayXf expected_weights_Xi(2); // 2 elements (x2 and c)    
    expected_weights_Xi << 0.2736814, 0.2736814;

    auto actual_weights_i = SS.terminal_weights.at(DataType::ArrayI);
    Eigen::Map<ArrayXf> actual_weights_Xi(actual_weights_i.data(), actual_weights_i.size());
    
    ASSERT_TRUE(expected_weights_Xi.isApprox(actual_weights_Xi));

    ArrayXf expected_weights_Xb(2); // 2 elements (x0 and c)    
    expected_weights_Xb << 0.8117065, 0.8117065;

    auto actual_weights_b = SS.terminal_weights.at(DataType::ArrayB);
    Eigen::Map<ArrayXf> actual_weights_Xb(actual_weights_b.data(), actual_weights_b.size());
    
    ASSERT_TRUE(expected_weights_Xb.isApprox(actual_weights_Xb));
}