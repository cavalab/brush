#include "testsHeader.h"
#include "../../src/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"

TEST(Operators, Mutation)
{
    // test mutation
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
            auto Child = PRG.mutate(SS);

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

TEST(Operators, Crossover)
{
    // test mutation
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
            RegressorProgram PRG1 = SS.make_regressor(d, s);
            RegressorProgram PRG2 = SS.make_regressor(d, s);

            fmt::print(
                "=================================================\n"
                "depth = {}, size= {}\n"
                "Initial Model 1: {}\n"
                "Initial Model 2: {}\n",
                d, s, 
                PRG1.get_model("compact", true),
                PRG2.get_model("compact", true)
            );
            PRG1.fit(data);
            PRG2.fit(data);
            ArrayXf y_pred = PRG1.predict(data);
            fmt::print("cross one\n");
            auto Child1 = PRG1.cross(PRG2);
            fmt::print(
                "Model 1 after cross: {}\n"
                "Model 2 after cross: {}\n",
                PRG1.get_model("compact", true),
                PRG2.get_model("compact", true)
            );
            fmt::print("cross two\n");
            auto Child2 = PRG2.cross(PRG1);

            fmt::print(
                "Crossed Model 1: {}\n"
                "Crossed Model 2: {}\n"
                "=================================================\n",
                Child1.get_model("compact", true),
                Child2.get_model("compact", true)
            );

            Child1.fit(data);
            Child2.fit(data);
            auto child_pred1 = Child1.predict(data);
            auto child_pred2 = Child2.predict(data);
        }
    }
}