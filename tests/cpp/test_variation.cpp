#include "testsHeader.h"
#include "../../src/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"

TEST(Operators, Mutation)
{
    // test mutation
    // TODO: set random seed

    PARAMS["mutation_options"] = {
        {"point",0.25}, {"insert", 0.25}, {"delete", 0.25}, {"toggle_weight", 0.25}
    };
    
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
        int successes = 0;
        for (int s = 1; s < 10; ++s)
        {
            fmt::print("d={},s={}\n",d,s);
            fmt::print("make_regressor\n");

            // if we set max_size and max_depth to zero, it will use the
            // values in the global PARAMS. Otherwise, it will respect the
            // values passed as argument.
            RegressorProgram PRG = SS.make_regressor(d, s);

            fmt::print("PRG.fit(data);\n");
            PRG.fit(data);
            ArrayXf y_pred = PRG.predict(data);
            
            // applying mutation and checking if the optional result is non-empty
            fmt::print("auto Child = PRG.mutate();\n");
            auto opt = PRG.mutate();

            if (!opt){
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Initial Model: {}\n"
                    "Mutation failed to create a child",
                    d, s, 
                    PRG.get_model("compact", true)
                );
            }
            else {
                successes += 1;
                auto Child = opt.value();
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Initial Model: {}\n"
                    "Mutated Model: {}\n",
                    d, s, 
                    PRG.get_model("compact", true),
                    Child.get_model("compact", true)
                );

                fmt::print("child fit\n");
                Child.fit(data);
                y_pred = Child.predict(data);
            }
        }
        // since x1 and x2 have same type, we shoudn't get fails
        ASSERT_TRUE(successes > 0);
    }
}

TEST(Operators, MutationSizeAndDepthLimit)
{
    PARAMS["mutation_options"] = {
        {"point",0.25}, {"insert", 0.25}, {"delete", 0.25}, {"toggle_weight", 0.25}
    };
        
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

    // split operator --> arity 3
    // prod operator  --> arity 4
    int max_arity = 4;

    for (int d = 5; d < 15; ++d)
    {
        int successes = 0;
        for (int s = 5; s < 15; ++s)
        {
            PARAMS["max_size"]  = s;
            PARAMS["max_depth"] = d;

            fmt::print("d={},s={}\n",d,s);
            fmt::print("make_regressor\n");

            // Enforcing that the parents does not exceed max_size by
            // taking into account the highest arity of the function nodes;
            // and the max_depth+1 that PTC2 can generate
            RegressorProgram PRG = SS.make_regressor(d-1, s - max_arity);
            
            auto PRG_model = PRG.get_model("compact", true);

            auto opt = PRG.mutate();

            if (!opt){
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Initial Model: {}\n"
                    "Mutation failed to create a child",
                    d, s, 
                    PRG.get_model("compact", true)
                );
            }
            else {
                successes += 1;
                
                // Extracting the child from the std::optional and checking
                // if it is within size and depth restrictions. There is no
                // margin for having slightly bigger expressions.
                auto Child = opt.value();
                
                fmt::print("print\n");
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Initial Model: {}\n"
                    "Mutated Model: {}\n"
                    "Mutated depth: {}\n"
                    "Mutated size : {}\n",
                    d, s, 
                    PRG.get_model("compact", true),
                    Child.get_model("compact", true),
                    Child.Tree.max_depth(),
                    Child.Tree.size()
                );

                // Original didn't change
                ASSERT_TRUE(PRG_model == PRG.get_model("compact", true));
                
                ASSERT_TRUE(Child.size() > 0);
                ASSERT_TRUE(Child.size() <= s);

                ASSERT_TRUE(Child.Tree.size() > 0);
                ASSERT_TRUE(Child.Tree.size() <= s);

                ASSERT_TRUE(Child.Tree.max_depth() >= 0);
                ASSERT_TRUE(Child.Tree.max_depth() <= d);
            }
        }
        ASSERT_TRUE(successes > 0);
    }
}

TEST(Operators, Crossover)
{
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
        int successes = 0;
        for (int s = 1; s < 10; ++s)
        {
            RegressorProgram PRG1 = SS.make_regressor(d, s);
            RegressorProgram PRG2 = SS.make_regressor(d, s);
            PRG1.fit(data);
            PRG2.fit(data);

            fmt::print(
                "=================================================\n"
                "depth = {}, size= {}\n"
                "Initial Model 1: {}\n"
                "Initial Model 2: {}\n",
                d, s, 
                PRG1.get_model("compact", true),
                PRG2.get_model("compact", true)
            );

            ArrayXf y_pred = PRG1.predict(data);
            fmt::print("cross one\n");

            auto opt = PRG1.cross(PRG2);
            if (!opt){
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Original model 1: {}\n"
                    "Original model 2: {}\n",
                    "Crossover failed to create a child",
                    d, s, 
                    PRG1.get_model("compact", true),
                    PRG2.get_model("compact", true)
                );
            }
            else {
                successes += 1;
                auto Child = opt.value();
                fmt::print(
                    "Original model 1 after cross: {}\n"
                    "Original model 2 after cross: {}\n",
                    PRG1.get_model("compact", true),
                    PRG2.get_model("compact", true)
                );
                fmt::print(
                    "Crossed Model: {}\n"
                    "=================================================\n",
                    Child.get_model("compact", true)
                );
                Child.fit(data);
                auto child_pred1 = Child.predict(data);
            }
        }
        ASSERT_TRUE(successes > 0);
    }
}

TEST(Operators, CrossoverSizeAndDepthLimit)
{
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

    // split operator --> arity 3
    // prod operator  --> arity 4
    int max_arity = 4;

    for (int d = 5; d < 15; ++d)
    {
        int successes = 0;
        for (int s = 5; s < 15; ++s)
        {
            PARAMS["max_size"]  = s;
            PARAMS["max_depth"] = d;

            // Enforcing that the parents does not exceed max_size by
            // taking into account the highest arity of the function nodes
            RegressorProgram PRG1 = SS.make_regressor(d-1, s-max_arity);
            RegressorProgram PRG2 = SS.make_regressor(d-1, s-max_arity);

            auto PRG1_model = PRG1.get_model("compact", true);
            auto PRG2_model = PRG2.get_model("compact", true);

            fmt::print(
                "=================================================\n"
                "depth = {}, size= {}\n"
                "Initial Model 1: {}\n"
                "Initial Model 2: {}\n",
                d, s, 
                PRG1.get_model("compact", true),
                PRG2.get_model("compact", true)
            );

            fmt::print("cross\n");
            auto opt = PRG1.cross(PRG2);

            if (!opt){
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Original model 1: {}\n"
                    "Original model 2: {}\n",
                    "Crossover failed to create a child",
                    d, s, 
                    PRG1.get_model("compact", true),
                    PRG2.get_model("compact", true)
                );
            }
            else {
                successes += 1;
                auto Child = opt.value();
                fmt::print(
                    "Child Model      : {}\n"
                    "Child Model depth: {}\n"
                    "Child Model size : {}\n"
                    "=================================================\n",
                    Child.get_model("compact", true),
                    Child.Tree.max_depth(), Child.Tree.size()
                );

                // Original didn't change
                ASSERT_TRUE(PRG1_model == PRG1.get_model("compact", true));
                ASSERT_TRUE(PRG2_model == PRG2.get_model("compact", true));

                // Child is within restrictions
                ASSERT_TRUE(Child.size() > 0);
                ASSERT_TRUE(Child.size() <= s);
                ASSERT_TRUE(Child.Tree.size() > 0);
                ASSERT_TRUE(Child.Tree.size() <= s);

                ASSERT_TRUE(Child.Tree.max_depth() >= 0);
                ASSERT_TRUE(Child.Tree.max_depth() <= d);
            }
        }
        ASSERT_TRUE(successes > 0);
    }
}