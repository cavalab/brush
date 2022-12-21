#include "testsHeader.h"
#include "../../src/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"

TEST(Program, MakeRegressor)
{
        
    Dataset data = Data::read_csv("examples/datasets/d_enc.csv","label");


    SearchSpace SS;
    SS.init(data);

    // Program<ArrayXf> DXtree;
    for (int d = 1; d < 10; ++d)
        for (int s = 1; s < 10; ++s)
        {
            RegressorProgram PRG = SS.make_regressor(d, s);
            fmt::print(
                "=================================================\n"
                "Tree model for depth = {}, size= {}: {}\n"
                "=================================================\n",
                d, s, PRG.get_model("compact", true)
            );
        }
}

TEST(Program, FitRegressor)
{
        
    Dataset data = Data::read_csv("examples/datasets/d_enc.csv","label");

    SearchSpace SS;
    SS.init(data);

    for (int t = 0; t < 10; ++t) {
        for (int d = 1; d < 10; ++d) { 
            for (int s = 1; s < 100; s+=10) {
                RegressorProgram PRG = SS.make_regressor(d, s);
                fmt::print(
                    "=================================================\n"
                    "Tree model for depth = {}, size= {}: {}\n"
                    "=================================================\n",
                    d, s, PRG.get_model("compact", true)
                );
                PRG.fit(data);
                auto y = PRG.predict(data);
            }
        }
    }
}

TEST(Program, FitClassifier)
{
        
    Dataset data = Data::read_csv("examples/datasets/d_analcatdata_aids.csv","target");
    SearchSpace SS;
    SS.init(data);

    for (int t = 0; t < 10; ++t) {
        for (int d = 1; d < 10; ++d) { 
            for (int s = 1; s < 100; s+=10) {
                auto PRG = SS.make_classifier(d, s);
                fmt::print(
                    "=================================================\n"
                    "Tree model for depth = {}, size= {}: {}\n"
                    "=================================================\n",
                    d, s, PRG.get_model("compact", true)
                );
                PRG.fit(data);
                auto y = PRG.predict(data);
                auto yproba = PRG.predict_proba(data);
            }
        }
    }
}



TEST(Program, Mutation)
{
    // test mutation
    // y = 3.14*X1 + 1.68*X2
    // r: random numbers
    // X1: cos(r)
    // X2: sin(r)
    /* r = 0.54340523, 0.98342536, 0.52725502, 0.1019157 , 0.13250716, */
    /*          0.2273736 , 0.78280196, 0.31946665, 0.07088554, 0.99791986; */
    // TODO: set random seed
    MatrixXf X(10,2);
    ArrayXf y(10);
    X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
         0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,
         0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
         0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;
    y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
             3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;
    Dataset data(X,y);

    SearchSpace SS;
    SS.init(data);

    for (int d = 1; d < 10; ++d)
    {
        for (int s = 1; s < 10; ++s)
        {
            RegressorProgram PRG = SS.make_regressor(d, s);
            PRG.fit(data);
            ArrayXf y_pred = PRG.predict(data);
            auto Child = PRG.mutate();

            fmt::print(
                "=================================================\n"
                "depth = {}, size= {}\n"
                "Initial Model: {}\n"
                "Mutated Model: {}\n",
                d, s, 
                PRG.get_model("compact", true),
                Child.get_model("compact", true)
            );

            Child.fit(data);
            y_pred = Child.predict(data);
        }
    }
}

TEST(Program, WeightOptimization)
{
    // TODO
    // // test backpropagation
    // // y = 3.14*X1 + 1.68*X2
    // // r: random numbers
    // // X1: cos(r)
    // // X2: sin(r)
    // /* r = 0.54340523, 0.98342536, 0.52725502, 0.1019157 , 0.13250716, */
    // /*          0.2273736 , 0.78280196, 0.31946665, 0.07088554, 0.99791986; */
    // MatrixXf X(10,2);
    // ArrayXf y(10);
    // X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
    //      0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,
    //      0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
    //      0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;
    // y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
    //          3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;
    // Dataset data(X,y);

    // unordered_map<string, float> user_ops = {
    //     {"Add", 1},
    //     {"Sub", 1}, 
    //     {"Mul", 0.5},
    //     {"Sin", 0.1},
    //     {"SplitOn", 0.5}, 
    //     {"SplitBest", 0.5} 
    // };
    // SearchSpace SS(data, user_ops);
    // /* SS.init(data); */

    // auto DXtree = SS.make_program<RegressorProgram>(9,9);
    // DXtree.fit(data);
    // ofstream file;
    // file.open(fmt::format("dx_model.dot"));
    // file <<  DXtree.get_model("dot", true);
    // file.close();
    // cout << "generating predictions\n";
    // ArrayXf y_pred = DXtree.predict(data);
    // cout << "gradient descent\n";
    // cout << "calculating loss\n";
    // cout << "y_pred: " << y_pred.transpose() << endl;
    // cout << "y: " << y.transpose() << endl;
    // cout << "loss: " << (y_pred - y).square().transpose() << endl;
    // ArrayXf d_loss = 2*(y_pred - y);
    // for (int i = 0; i < 20; ++i)
    // {
    //     DXtree.grad_descent(d_loss, data);
    //     y_pred = DXtree.predict(data);
    //     cout << "updated y_pred: " << y_pred.transpose() << endl;
    //     cout << "             y: " << y.transpose() << endl;
    //     cout << "loss: " << (y_pred - y).square().transpose() << endl;
    //     d_loss = 2*(y_pred - y);
    // }

}