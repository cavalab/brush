#include "testsHeader.h"
#include "../../src/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"

TEST(Operators, Mutation)
{
    PARAMS["mutation_options"] = {
        {"point",0.25}, {"insert", 0.25}, {"delete", 0.25}, {"toggle_weight", 0.25}
    };
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
            fmt::print("d={},s={}\n",d,s);
            fmt::print("make_regressor\n");
            RegressorProgram PRG = SS.make_regressor(d, s);
            fmt::print("PRG.fit(data);\n");
            PRG.fit(data);
            ArrayXf y_pred = PRG.predict(data);
            fmt::print("auto Child = PRG.mutate(SS);\n");
            auto Child = PRG.mutate();

            fmt::print("print\n");
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

            auto Child = PRG.mutate();

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
            
            // Child is within restrictions. Here we expect the generated
            // expression to have at most max_size nodes (there is no tolerance 
            // gap as PTC2 has). Notice that this is only valid if the original
            // parent is already respecting the max_size
            ASSERT_TRUE(Child.size() > 0);
            ASSERT_TRUE(Child.size() <= s);

            ASSERT_TRUE(Child.Tree.size() > 0);
            ASSERT_TRUE(Child.Tree.size() <= s);

            ASSERT_TRUE(Child.Tree.max_depth() >= 0);
            ASSERT_TRUE(Child.Tree.max_depth() <= d);
        }
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
                "Crossed Model 1      : {}\n"
                "Crossed Model 1 depth: {}\n"
                "Crossed Model 1 size : {}\n"
                "Crossed Model 2      : {}\n"
                "Crossed Model 2 depth: {}\n"
                "Crossed Model 2 size : {}\n"
                "=================================================\n",
                Child1.get_model("compact", true),
                Child1.Tree.max_depth(), Child1.Tree.size(), 
                Child2.get_model("compact", true),
                Child2.Tree.max_depth(), Child2.Tree.size()
            );

            // Original didn't change
            ASSERT_TRUE(PRG1_model == PRG1.get_model("compact", true));
            ASSERT_TRUE(PRG2_model == PRG2.get_model("compact", true));

            // Child1 is within restrictions
            ASSERT_TRUE(Child1.size() > 0);
            ASSERT_TRUE(Child1.size() <= s);
            ASSERT_TRUE(Child1.Tree.size() > 0);
            ASSERT_TRUE(Child1.Tree.size() <= s);

            ASSERT_TRUE(Child1.Tree.max_depth() >= 0);
            ASSERT_TRUE(Child1.Tree.max_depth() <= d);

            // Child2 is within restrictions
            ASSERT_TRUE(Child2.size() > 0);
            ASSERT_TRUE(Child2.size() <= s);
            ASSERT_TRUE(Child2.Tree.size() > 0);
            ASSERT_TRUE(Child2.Tree.size() <= s);

            ASSERT_TRUE(Child2.Tree.max_depth() >= 0);
            ASSERT_TRUE(Child2.Tree.max_depth() <= d);
        }
    }
}

TEST(Operators, MutationSizeAndDepthPARAMS)
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

    for (int d = 1; d < 10; ++d)
    {
        for (int s = 1; s < 10; ++s)
        {
            PARAMS["max_size"]  = s;
            PARAMS["max_depth"] = d;

            fmt::print("d={},s={}\n",d,s);
            fmt::print("make_regressor\n");

            RegressorProgram PRG = SS.make_regressor(0, 0);
            
            auto PRG_model = PRG.get_model("compact", true);

            auto Child = PRG.mutate();

            // Child is within restrictions. Here we allow the mutation
            // to generate slightly bigger expressions (because the original
            // parents can also have this offset due to PTC2 generation method)
            ASSERT_TRUE(Child.size() > 0);
            ASSERT_TRUE(Child.size() <= s+max_arity);

            ASSERT_TRUE(Child.Tree.size() > 0);
            ASSERT_TRUE(Child.Tree.size() <= s+max_arity);

            ASSERT_TRUE(Child.Tree.max_depth() >= 0);
            ASSERT_TRUE(Child.Tree.max_depth() <= d+1);
        }
    }
}

TEST(Operators, CrossoverSizeAndDepthPARAMS)
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

    for (int d = 1; d < 10; ++d)
    {
        for (int s = 1; s < 10; ++s)
        {
            PARAMS["max_size"]  = s;
            PARAMS["max_depth"] = d;

            RegressorProgram PRG1 = SS.make_regressor(0, 0);
            RegressorProgram PRG2 = SS.make_regressor(0, 0);

            auto PRG1_model = PRG1.get_model("compact", true);
            auto PRG2_model = PRG2.get_model("compact", true);

            auto Child1 = PRG1.cross(PRG2);
            auto Child2 = PRG2.cross(PRG1);

            // Child1 is within restrictions
            ASSERT_TRUE(Child1.size() > 0);
            ASSERT_TRUE(Child1.size() <= s+max_arity);
            ASSERT_TRUE(Child1.Tree.size() > 0);
            ASSERT_TRUE(Child1.Tree.size() <= s+max_arity);

            ASSERT_TRUE(Child1.Tree.max_depth() >= 0);
            ASSERT_TRUE(Child1.Tree.max_depth() <= d+1);

            // Child2 is within restrictions
            ASSERT_TRUE(Child2.size() > 0);
            ASSERT_TRUE(Child2.size() <= s+max_arity);
            ASSERT_TRUE(Child2.Tree.size() > 0);
            ASSERT_TRUE(Child2.Tree.size() <= s+max_arity);

            ASSERT_TRUE(Child2.Tree.max_depth() >= 0);
            ASSERT_TRUE(Child2.Tree.max_depth() <= d+1);
        }
    }
}
