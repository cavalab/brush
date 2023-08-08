#include "testsHeader.h"
#include "../../src/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"

TEST(SearchSpace, Initialization)
{
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
}