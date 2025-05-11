#include "testsHeader.h"
// #include "../../src/bandit/bandit.cpp"

TEST(Variation, FixedRootDoesntChange)
{
    Parameters params;

    MatrixXf X(10,2);
    ArrayXf y(10);
    X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
         0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

         0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
         0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

    y << 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    Dataset data(X,y);

    SearchSpace SS;
    SS.init(data);

    auto logistic_hash = Signature<ArrayXf(ArrayXf)>().hash();

    // TODO: use these values for d and s in all tests (not 1, 1 for example)
    for (int d = 3; d < 6; ++d)
    {
        for (int s = 10; s < 50; ++s) 
        {
            params.max_size  = s;
            params.max_depth = d;

            Variation variator = Variation<ProgramType::BinaryClassifier>(params, SS, data);

            int successes = 0;
            for (int attempt = 0; attempt < 50; ++attempt)
            {
                // different program types changes how predict works (and the rettype of predict)
                ClassifierProgram PRG = SS.make_classifier(0, 0, params);
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Initial Model 1: {}\n",
                    d, s, 
                    PRG.get_model("compact", true)
                );

                Node root = *(PRG.Tree.begin());

                ASSERT_TRUE(root.node_type == NodeType::Logistic);
                ASSERT_TRUE(root.ret_type == DataType::ArrayF);
                ASSERT_TRUE(root.sig_hash == logistic_hash);
                ASSERT_TRUE(root.get_prob_change()==0.0);
                ASSERT_TRUE(root.fixed==true);

                Individual<PT::BinaryClassifier> IND(PRG);
                auto opt_mutation = variator.mutate(IND);
                
                if (opt_mutation)
                {
                    successes += 1;
                    auto Mut_Child = opt_mutation.value();
                    fmt::print("After mutation : {}\n",
                               Mut_Child.program.get_model("compact", true));

                    Node mut_child_root = *(Mut_Child.program.Tree.begin());

                    ASSERT_TRUE(mut_child_root.node_type == NodeType::Logistic);
                    ASSERT_TRUE(mut_child_root.ret_type == DataType::ArrayF);
                    ASSERT_TRUE(mut_child_root.sig_hash == logistic_hash);
                    ASSERT_TRUE(mut_child_root.get_prob_change()==0.0);
                    ASSERT_TRUE(mut_child_root.fixed==true);
                }

                ClassifierProgram PRG2 = SS.make_classifier(0, 0, params);

                Individual<PT::BinaryClassifier> IND2(PRG2);
                auto opt_cx = variator.cross(IND, IND2);

                if (opt_cx)
                {
                    successes += 1;
                    auto CX_Child = opt_cx.value();
                    fmt::print("After crossover: {}\n",
                               CX_Child.program.get_model("compact", true));

                    Node cx_child_root = *(CX_Child.program.Tree.begin());

                    ASSERT_TRUE(cx_child_root.node_type == NodeType::Logistic);
                    ASSERT_TRUE(cx_child_root.ret_type == DataType::ArrayF);
                    ASSERT_TRUE(cx_child_root.sig_hash == logistic_hash);
                    ASSERT_TRUE(cx_child_root.get_prob_change()==0.0);
                    ASSERT_TRUE(cx_child_root.fixed==true);
                }

                // root remained unchanged
                ASSERT_TRUE(root.node_type == NodeType::Logistic);
                ASSERT_TRUE(root.ret_type == DataType::ArrayF);
                ASSERT_TRUE(root.sig_hash == logistic_hash);
                ASSERT_TRUE(root.get_prob_change()==0.0);
                ASSERT_TRUE(root.fixed==true);
            }
            ASSERT_TRUE(successes > 0);
        }
    }
}

TEST(Variation, InsertMutationWorks)
{
    // TODO: this tests could be parameterized (one type of mutation each).
    // To understand design implementation of this test, check Mutation test

    Parameters params;
    params.mutation_probs = {
        {"point", 0.0},
        {"insert", 1.0},
        {"delete", 0.0},
        {"subtree", 0.0},
        {"toggle_weight_on", 0.0},
        {"toggle_weight_off", 0.0}
    };

    // retrieving the options to check if everything was set right
    std::cout << "Initial mutation configuration" << std::endl;
    for (const auto& [k, v] : params.mutation_probs)
        std::cout << k << " : " << v << std::endl;

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

    Variation variator = Variation<ProgramType::Regressor>(params, SS, data);

    int successes = 0;
    for (int attempt = 0; attempt < 100; ++attempt)
    {        
        params.max_size  = 50;
        params.max_depth = 6;

        fmt::print("d={},s={}\n", params.max_depth, params.max_size);
        fmt::print("make_regressor\n");

        // creating a "small" program (with a plenty amount of space to insert stuff)
        RegressorProgram PRG = SS.make_regressor(5, 5, params);

        fmt::print("PRG.fit(data);\n");
        PRG.fit(data);
        ArrayXf y_pred = PRG.predict(data);
        
        // applying mutation and checking if the optional result is non-empty
        fmt::print("auto Child = PRG.mutate();\n");

        // We should assume that it will be always the insert mutation

        Individual<PT::Regressor> IND(PRG);

        auto opt = variator.mutate(IND); 

        if (opt){
            successes += 1;
            auto Child = opt.value();
            fmt::print(
                "=================================================\n"
                "depth = {}, size= {}\n"
                "Initial Model: {}\n"
                "Mutated Model: {}\n",
                params.max_depth, params.max_size,
                IND.program.get_model("compact", true),
                Child.program.get_model("compact", true)
            );

            fmt::print("child fit\n");
            Child.fit(data);
            y_pred = Child.predict(data);

            // since we successfully inserted a node, this should be always true
            ASSERT_TRUE(Child.program.size() > IND.program.size());

            // maybe the insertion spot was a shorter branch than the maximum
            // depth. At least, xmen depth should be equal to its parent
            ASSERT_TRUE(Child.program.depth() >= IND.program.depth());
        }

        // lets also see if it always fails when the child exceeds the maximum limits
        variator.parameters.set_max_depth(IND.program.depth());
        variator.parameters.set_max_size(IND.program.size());

        auto opt2 = variator.mutate(IND);
        if (opt2){ // This shoudl't happen. We'll print the error
            auto Child2 = opt2.value();

            std::cout << "Fail failed. Mutation weights:" << std::endl;
            for (const auto& [k, v] : params.mutation_probs)
                std::cout << k << " : " << v << std::endl;

            fmt::print(
                "max depth = {}, max size= {}\n"
                "Initial Model: {}\n"
                "Mutated Model: {}\n"
                "=================================================\n",
                params.max_depth, params.max_size,
                IND.program.get_model("compact", true),
                Child2.program.get_model("compact", true)
            );
            ASSERT_TRUE(opt2==std::nullopt); // this will fail, so we can see the log
        }
    }
    ASSERT_TRUE(successes > 0);
}

TEST(Variation, Mutation)
{
    Parameters params;
    
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

    int successes = 0;
    for (int d = 1; d < 6; ++d)
    {
        for (int s = 10; s < 20; ++s)
        {
            params.max_size  = s;
            params.max_depth = d;

            Variation variator = Variation<ProgramType::Regressor>(params, SS, data);

            fmt::print("d={},s={}\n",d,s);
            fmt::print("make_regressor\n");

            // if we set max_size and max_depth to zero, it will use the
            // values in the global PARAMS. Otherwise, it will respect the
            // values passed as argument.
            RegressorProgram PRG = SS.make_regressor(0, 0, params);

            fmt::print("PRG.fit(data);\n");
            PRG.fit(data);

            // saving a string representation
            auto PRG_model = PRG.get_model("compact", true);

            fmt::print(
                "=================================================\n"
                "Original model (BEFORE MUTATION) 1: {}\n",
                PRG.get_model("compact", true)
            );
            ArrayXf y_pred = PRG.predict(data);
            
            // applying mutation and checking if the optional result is non-empty
            fmt::print("auto Child = PRG.mutate();\n");

            Individual<PT::Regressor> IND(PRG);
            auto opt = variator.mutate(IND);

            if (!opt){
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Initial Model: {}\n"
                    "Mutation failed to create a child",
                    d, s, 
                    IND.program.get_model("compact", true)
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
                    IND.program.get_model("compact", true),
                    Child.program.get_model("compact", true)
                );

                fmt::print("child fit\n");
                Child.fit(data);
                y_pred = Child.predict(data);

                // no collateral effect (parent still the same)
                ASSERT_TRUE(PRG_model == IND.program.get_model("compact", true));
            }
        }
    }
    // since x1 and x2 have same type, we shoudn't get fails
    ASSERT_TRUE(successes > 0);
}

TEST(Variation, MutationSizeAndDepthLimit)
{
    Parameters params;
        
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
    
    // prod operator  --> arity 4: prod(T1, T2, T3)
    // split best     --> arity 6: if(terminal > value, T_case_true, T_case_false)
    int max_arity = 6;

    int successes = 0;
    for (int d = 1; d < 6; ++d)
    {
        for (int s = 5; s < 15; ++s)
        {
            params.max_size  = s;
            params.max_depth = d;
            
            // creating and fitting a child
            Variation variator = Variation<ProgramType::Regressor>(params, SS, data);

            fmt::print("d={},s={}\n",d,s);
            fmt::print("make_regressor\n");

            // Enforcing that the parents does not exceed max_size by
            // taking into account the highest arity of the function nodes;
            // and the max_depth+1 that PTC2 can generate
            RegressorProgram PRG = SS.make_regressor(0, 0, params);
            
            auto PRG_model = PRG.get_model("compact", true);

            Individual<PT::Regressor> IND(PRG);
            auto opt = variator.mutate(IND);

            if (!opt){
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Initial Model: {}\n"
                    "Mutation failed to create a child",
                    d, s, 
                    IND.program.get_model("compact", true)
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
                    IND.program.get_model("compact", true),
                    Child.program.get_model("compact", true),
                    Child.program.depth(),
                    Child.program.size()
                );

                // Original didn't change
                ASSERT_TRUE(PRG_model == IND.program.get_model("compact", true));
                
                ASSERT_TRUE(Child.program.size() > 0);
                ASSERT_TRUE(Child.program.size() <= s);

                ASSERT_TRUE(Child.program.size() > 0);
                ASSERT_TRUE(Child.program.size() <= s);

                ASSERT_TRUE(Child.program.depth() >= 0);
                ASSERT_TRUE(Child.program.depth() <= d);
            }
        }
    }
    ASSERT_TRUE(successes > 0);
}

TEST(Variation, Crossover)
{
    Parameters params;

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

    int successes = 0;
    for (int d = 2; d < 6; ++d)
    {
        for (int s = 5; s < 15; ++s)
        {
            params.max_size  = s;
            params.max_depth = d;
            Variation variator = Variation<ProgramType::Regressor>(params, SS, data);
            
            RegressorProgram PRG1 = SS.make_regressor(d, 0, params);
            PRG1.fit(data);
            auto PRG1_model = PRG1.get_model("compact", true);

            RegressorProgram PRG2 = SS.make_regressor(d, 0, params);
            PRG2.fit(data);
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

            ArrayXf y_pred = PRG1.predict(data);
            fmt::print("cross one\n");

            Individual<PT::Regressor> IND1(PRG1);
            Individual<PT::Regressor> IND2(PRG2);
            auto opt = variator.cross(IND1, IND2);

            if (!opt){
                fmt::print(
                    "=================================================\n"
                    "depth = {}, size= {}\n"
                    "Original model 1: {}\n"
                    "Original model 2: {}\n",
                    "Crossover failed to create a child",
                    d, s, 
                    IND1.program.get_model("compact", true),
                    IND2.program.get_model("compact", true)
                );
            }
            else {
                successes += 1;
                auto Child = opt.value();
                fmt::print(
                    "Original model 1 after cross: {}\n"
                    "Original model 2 after cross: {}\n",
                    IND1.program.get_model("compact", true),
                    IND2.program.get_model("compact", true)
                );
                fmt::print(
                    "Crossed Model: {}\n"
                    "=================================================\n",
                    Child.program.get_model("compact", true)
                );
                Child.fit(data);
                auto child_pred1 = Child.predict(data);

                // no collateral effect (parent still the same)
                ASSERT_TRUE(PRG1_model == IND1.program.get_model("compact", true));
                ASSERT_TRUE(PRG2_model == IND2.program.get_model("compact", true));
            }
        }
    }
    ASSERT_TRUE(successes > 0);
}

TEST(Variation, CrossoverSizeAndDepthLimit)
{
    Parameters params;

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

    // prod operator  --> arity 4: prod(T1, T2, T3)
    // split best     --> arity 6: if(terminal > value, T_case_true, T_case_false)
    int max_arity = 6;

    int successes = 0;
    for (int d = 1; d < 6; ++d)
    {
        for (int s = 5; s < 15; ++s)
        {
            params.max_size  = s;
            params.max_depth = d;
            Variation variator = Variation<ProgramType::Regressor>(params, SS, data);

            // Enforcing that the parents does not exceed max_size by
            // taking into account the highest arity of the function nodes
            RegressorProgram PRG1 = SS.make_regressor(0, 0, params);
            RegressorProgram PRG2 = SS.make_regressor(0, 0, params);

            auto PRG1_model = PRG1.get_model("compact", true);
            auto PRG2_model = PRG2.get_model("compact", true);

            fmt::print(
                "=================================================\n"
                "settings: depth = {}, size= {}\n"
                "Original model 1: {}\n"
                "depth = {}, size= {}\n"
                "Original model 2: {}\n"
                "depth = {}, size= {}\n",
                d, s, 
                PRG1.get_model("compact", true),
                PRG1.depth(), PRG1.size(),
                PRG2.get_model("compact", true),
                PRG2.depth(), PRG2.size()
            );

            fmt::print("cross\n");
            Individual<PT::Regressor> IND1(PRG1);
            Individual<PT::Regressor> IND2(PRG2);
            auto opt = variator.cross(IND1, IND2);

            if (!opt){
                fmt::print("Crossover failed to create a child"
                    "=================================================\n");
            }
            else {
                successes += 1;
                auto Child = opt.value();
                fmt::print(
                    "Child Model      : {}\n"
                    "Child Model depth: {}\n"
                    "Child Model size : {}\n"
                    "=================================================\n",
                    Child.program.get_model("compact", true),
                    Child.program.depth(), Child.program.size()
                );

                // Original didn't change
                ASSERT_TRUE(PRG1_model == IND1.program.get_model("compact", true));
                ASSERT_TRUE(PRG2_model == IND2.program.get_model("compact", true));

                // Child is within restrictions
                ASSERT_TRUE(Child.program.size() > 0);
                ASSERT_TRUE(Child.program.size() <= s + 3*max_arity);

                ASSERT_TRUE(Child.program.depth() >= 0);
                ASSERT_TRUE(Child.program.depth() <= d);
            }
        }
    }
    ASSERT_TRUE(successes > 0);
}