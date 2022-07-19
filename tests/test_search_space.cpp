#include "testsHeader.h"
#include "../src/search_space.h"
#include "../src/program.h"

TEST(SearchSpace, Initialization)
{
	
    MatrixXf X(4,2); 
    MatrixXf X_v(3,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    X_v <<  0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    X_v.transposeInPlace();

    ArrayXf y(4); 
    ArrayXf y_v(3); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;
    y_v << 0.57015434, -1.20648656, -2.68773747;
    
	Data dt(X, y);
    Data dv(X_v, y_v);
    
    map<string, float> user_ops = {
        {"Add", 1},
        {"Sub", 1},
        {"Div", .5},
        {"Times", 0.5}
    };

    // SearchSpace SS;
    SearchSpace SS;
    SS.init(dt, user_ops);
}
