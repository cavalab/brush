// #include "testsHeader.h"
// #include "../../src/search_space.h"
// #include "../../src/program/program.h"
// #include "../../src/program/dispatch_table.h"
// #include "../../src/data/io.h"

// TEST(Variation, FixedRootDoesntChange)
// {
//     PARAMS["mutation_options"] = {
//         {"point",0.167}, {"insert", 0.167}, {"delete", 0.167}, {"subtree", 0.167}, {"toggle_weight_on", 0.167}, {"toggle_weight_off", 0.167}
//     };
//     PARAMS["max_size"]  = 20;
//     PARAMS["max_depth"] = 10;

//     MatrixXf X(10,2);
//     ArrayXf y(10);
//     X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
//          0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

//          0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
//          0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

//     y << 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;

//     Dataset data(X,y);

//     SearchSpace SS;
//     SS.init(data);

//     auto logistic_hash = Signature<ArrayXf(ArrayXf)>().hash();

//     for (int d = 1; d < 10; ++d)
//     {
//         for (int s = 1; s < 10; ++s) 
//         {
//             int successes = 0;
//             for (int attempt = 0; attempt < 10; ++attempt)
//             {
//                 // different program types changes how predict works (and the rettype of predict)
//                 ClassifierProgram PRG = SS.make_classifier(d, s);
//                 fmt::print(
//                     "=================================================\n"
//                     "depth = {}, size= {}\n"
//                     "Initial Model 1: {}\n",
//                     d, s, 
//                     PRG.get_model("compact", true)
//                 );

//                 Node root = *(PRG.Tree.begin());
//                 ASSERT_TRUE(root.node_type == NodeType::Logistic);
//                 ASSERT_TRUE(root.ret_type == DataType::ArrayF);
//                 ASSERT_TRUE(root.sig_hash == logistic_hash);
//                 ASSERT_TRUE(root.get_prob_change()==0.0);
//                 ASSERT_TRUE(root.fixed==true);

//                 auto opt_mutation = PRG.mutate();
//                 if (opt_mutation)
//                 {
//                     successes += 1;
//                     auto Mut_Child = opt_mutation.value();
//                     fmt::print("After mutation : {}\n",
//                                Mut_Child.get_model("compact", true));

//                     Node mut_child_root = *(Mut_Child.Tree.begin());
//                     ASSERT_TRUE(mut_child_root.node_type == NodeType::Logistic);
//                     ASSERT_TRUE(mut_child_root.ret_type == DataType::ArrayF);
//                     ASSERT_TRUE(mut_child_root.sig_hash == logistic_hash);
//                     ASSERT_TRUE(mut_child_root.get_prob_change()==0.0);
//                     ASSERT_TRUE(mut_child_root.fixed==true);
//                 }

//                 ClassifierProgram PRG2 = SS.make_classifier(d, s);
//                 auto opt_cx = PRG.cross(PRG2);
//                 if (opt_cx)
//                 {
//                     successes += 1;
//                     auto CX_Child = opt_cx.value();
//                     fmt::print("After crossover: {}\n",
//                                CX_Child.get_model("compact", true));

//                     Node cx_child_root = *(CX_Child.Tree.begin());
//                     ASSERT_TRUE(cx_child_root.node_type == NodeType::Logistic);
//                     ASSERT_TRUE(cx_child_root.ret_type == DataType::ArrayF);
//                     ASSERT_TRUE(cx_child_root.sig_hash == logistic_hash);
//                     ASSERT_TRUE(cx_child_root.get_prob_change()==0.0);
//                     ASSERT_TRUE(cx_child_root.fixed==true);
//                 }

//                 // root remained unchanged
//                 ASSERT_TRUE(root.node_type == NodeType::Logistic);
//                 ASSERT_TRUE(root.ret_type == DataType::ArrayF);
//                 ASSERT_TRUE(root.sig_hash == logistic_hash);
//                 ASSERT_TRUE(root.get_prob_change()==0.0);
//                 ASSERT_TRUE(root.fixed==true);
//             }
//             ASSERT_TRUE(successes > 0);
//         }
//     }
// }

// TEST(Variation, InsertMutationWorks)
// {
//     // TODO: this tests could be parameterized.
//     // To understand design implementation of this test, check Mutation test

//     PARAMS["mutation_options"] = {
//         {"point", 0.0}, {"insert", 1.0}, {"delete", 0.0}, {"subtree", 0.0}, {"toggle_weight_on", 0.0}, {"toggle_weight_off", 0.0}
//     };

//     // retrieving the options to check if everything was set right
//     std::cout << "Initial mutation configuration" << std::endl;
//     auto options = PARAMS["mutation_options"].get<std::map<string,float>>();
//     for (const auto& [k, v] : options)
//         std::cout << k << " : " << v << std::endl;

//     MatrixXf X(10,2);
//     ArrayXf y(10);
//     X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
//          0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

//          0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
//          0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

//     y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
//          3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;

//     Dataset data(X,y);

//     SearchSpace SS;
//     SS.init(data);

//     int successes = 0;
//     for (int attempt = 0; attempt < 100; ++attempt)
//     {
//         // we need to have big values here so the mutation will work
//         // (when the xmen child exceeds the maximum limits, mutation returns
//         // std::nullopt)
//         PARAMS["max_size"]  = 20;
//         PARAMS["max_depth"] = 10;
        
//         fmt::print("d={},s={}\n", PARAMS["max_depth"].get<int>(), PARAMS["max_size"].get<int>());
//         fmt::print("make_regressor\n");

//         // creating a "small" program (with a plenty amount of space to insert stuff)
//         RegressorProgram PRG = SS.make_regressor(5, 5);

//         fmt::print("PRG.fit(data);\n");
//         PRG.fit(data);
//         ArrayXf y_pred = PRG.predict(data);
        
//         // applying mutation and checking if the optional result is non-empty
//         fmt::print("auto Child = PRG.mutate();\n");
//         auto opt = PRG.mutate(); // We should assume that it will be always the insert mutation

//         if (opt){
//             successes += 1;
//             auto Child = opt.value();
//             fmt::print(
//                 "=================================================\n"
//                 "depth = {}, size= {}\n"
//                 "Initial Model: {}\n"
//                 "Mutated Model: {}\n",
//                 PARAMS["max_depth"].get<int>(), PARAMS["max_size"].get<int>(),
//                 PRG.get_model("compact", true),
//                 Child.get_model("compact", true)
//             );

//             fmt::print("child fit\n");
//             Child.fit(data);
//             y_pred = Child.predict(data);

//             // since we successfully inserted a node, this should be always true
//             ASSERT_TRUE(Child.size() > PRG.size());

//             // maybe the insertion spot was a shorter branch than the maximum
//             // depth. At least, xmen depth should be equal to its parent
//             ASSERT_TRUE(Child.depth() >= PRG.depth());
//         }

//         // lets also see if it always fails when the child exceeds the maximum limits
//         PARAMS["max_size"]  = PRG.size();
//         PARAMS["max_depth"] = PRG.depth();

//         auto opt2 = PRG.mutate();
//         if (opt2){ // This shoudl't happen. We'll print then error
//             auto Child2 = opt2.value();

//             std::cout << "Fail failed. Mutation weights:" << std::endl;
//             auto options2 = PARAMS["mutation_options"].get<std::map<string,float>>();
//             for (const auto& [k, v] : options2)
//                 std::cout << k << " : " << v << std::endl;

//             fmt::print(
//                 "=================================================\n"
//                 "depth = {}, size= {}\n"
//                 "Initial Model: {}\n"
//                 "Mutated Model: {}\n",
//                 PARAMS["max_depth"].get<int>(), PARAMS["max_size"].get<int>(),
//                 PRG.get_model("compact", true),
//                 Child2.get_model("compact", true)
//             );
//             ASSERT_TRUE(opt2==std::nullopt);
//         }
//     }
//     ASSERT_TRUE(successes > 0);
// }

// TEST(Variation, Mutation)
// {
//     PARAMS["write_mutation_trace"] = true;
//     PARAMS["mutation_options"] = {
//         {"point",0.167}, {"insert", 0.167}, {"delete", 0.167}, {"subtree", 0.167}, {"toggle_weight_on", 0.167}, {"toggle_weight_off", 0.167}
//     };
    
//     MatrixXf X(10,2);
//     ArrayXf y(10);
//     X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
//          0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

//          0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
//          0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

//     y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
//          3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;

//     Dataset data(X,y);

//     SearchSpace SS;
//     SS.init(data);

//     int successes = 0;
//     for (int d = 1; d < 10; ++d)
//     {
//         for (int s = 1; s < 10; ++s)
//         {
//             fmt::print("d={},s={}\n",d,s);
//             fmt::print("make_regressor\n");

//             // if we set max_size and max_depth to zero, it will use the
//             // values in the global PARAMS. Otherwise, it will respect the
//             // values passed as argument.
//             RegressorProgram PRG = SS.make_regressor(d, s);

//             fmt::print("PRG.fit(data);\n");
//             PRG.fit(data);
//             ArrayXf y_pred = PRG.predict(data);
            
//             // applying mutation and checking if the optional result is non-empty
//             fmt::print("auto Child = PRG.mutate();\n");
//             auto opt = PRG.mutate();

//             if (!opt){
//                 fmt::print(
//                     "=================================================\n"
//                     "depth = {}, size= {}\n"
//                     "Initial Model: {}\n"
//                     "Mutation failed to create a child",
//                     d, s, 
//                     PRG.get_model("compact", true)
//                 );
//                 fmt::print("{}", PARAMS["mutation_trace"].get<json>().dump());
//             }
//             else {
//                 successes += 1;
//                 auto Child = opt.value();
//                 fmt::print(
//                     "=================================================\n"
//                     "depth = {}, size= {}\n"
//                     "Initial Model: {}\n"
//                     "Mutated Model: {}\n",
//                     d, s, 
//                     PRG.get_model("compact", true),
//                     Child.get_model("compact", true)
//                 );

//                 fmt::print("child fit\n");
//                 Child.fit(data);
//                 y_pred = Child.predict(data);
//             }
//         }
//     }
//     // since x1 and x2 have same type, we shoudn't get fails
//     ASSERT_TRUE(successes > 0);
// }

// TEST(Variation, MutationSizeAndDepthLimit)
// {
//     PARAMS["write_mutation_trace"] = true;
//     PARAMS["mutation_options"] = {
//         {"point",0.167}, {"insert", 0.167}, {"delete", 0.167}, {"subtree", 0.167}, {"toggle_weight_on", 0.167}, {"toggle_weight_off", 0.167}
//     };
        
//     MatrixXf X(10,2);
//     ArrayXf y(10);
//     X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
//          0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

//          0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
//          0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

//     y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
//          3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;

//     Dataset data(X,y);

//     SearchSpace SS;
//     SS.init(data);
    
//     // prod operator  --> arity 4: prod(T1, T2, T3)
//     // split best     --> arity 6: if(terminal > value, T_case_true, T_case_false)
//     int max_arity = 6;

//     int successes = 0;
//     for (int d = 5; d < 15; ++d)
//     {
//         for (int s = 5; s < 15; ++s)
//         {
//             PARAMS["max_size"]  = s;
//             PARAMS["max_depth"] = d;

//             fmt::print("d={},s={}\n",d,s);
//             fmt::print("make_regressor\n");

//             // Enforcing that the parents does not exceed max_size by
//             // taking into account the highest arity of the function nodes;
//             // and the max_depth+1 that PTC2 can generate
//             RegressorProgram PRG = SS.make_regressor(d-1, s - max_arity);
            
//             auto PRG_model = PRG.get_model("compact", true);

//             auto opt = PRG.mutate();

//             if (!opt){
//                 fmt::print(
//                     "=================================================\n"
//                     "depth = {}, size= {}\n"
//                     "Initial Model: {}\n"
//                     "Mutation failed to create a child",
//                     d, s, 
//                     PRG.get_model("compact", true)
//                 );
//                 fmt::print("{}", PARAMS["mutation_trace"].get<json>().dump());
//             }
//             else {
//                 successes += 1;
                
//                 // Extracting the child from the std::optional and checking
//                 // if it is within size and depth restrictions. There is no
//                 // margin for having slightly bigger expressions.
//                 auto Child = opt.value();
                
//                 fmt::print("print\n");
//                 fmt::print(
//                     "=================================================\n"
//                     "depth = {}, size= {}\n"
//                     "Initial Model: {}\n"
//                     "Mutated Model: {}\n"
//                     "Mutated depth: {}\n"
//                     "Mutated size : {}\n",
//                     d, s, 
//                     PRG.get_model("compact", true),
//                     Child.get_model("compact", true),
//                     Child.depth(),
//                     Child.size()
//                 );

//                 // Original didn't change
//                 ASSERT_TRUE(PRG_model == PRG.get_model("compact", true));
                
//                 ASSERT_TRUE(Child.size() > 0);
//                 ASSERT_TRUE(Child.size() <= s);

//                 ASSERT_TRUE(Child.size() > 0);
//                 ASSERT_TRUE(Child.size() <= s);

//                 ASSERT_TRUE(Child.depth() >= 0);
//                 ASSERT_TRUE(Child.depth() <= d);
//             }
//         }
//     }
//     ASSERT_TRUE(successes > 0);
// }

// TEST(Variation, Crossover)
// {
//     MatrixXf X(10,2);
//     ArrayXf y(10);
//     X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
//          0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

//          0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
//          0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

//     y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
//          3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;

//     Dataset data(X,y);

//     SearchSpace SS;
//     SS.init(data);

//     int successes = 0;
//     for (int d = 1; d < 10; ++d)
//     {
//         for (int s = 1; s < 10; ++s)
//         {
//             RegressorProgram PRG1 = SS.make_regressor(d, s);
//             RegressorProgram PRG2 = SS.make_regressor(d, s);
//             PRG1.fit(data);
//             PRG2.fit(data);

//             fmt::print(
//                 "=================================================\n"
//                 "depth = {}, size= {}\n"
//                 "Initial Model 1: {}\n"
//                 "Initial Model 2: {}\n",
//                 d, s, 
//                 PRG1.get_model("compact", true),
//                 PRG2.get_model("compact", true)
//             );

//             ArrayXf y_pred = PRG1.predict(data);
//             fmt::print("cross one\n");

//             auto opt = PRG1.cross(PRG2);
//             if (!opt){
//                 fmt::print(
//                     "=================================================\n"
//                     "depth = {}, size= {}\n"
//                     "Original model 1: {}\n"
//                     "Original model 2: {}\n",
//                     "Crossover failed to create a child",
//                     d, s, 
//                     PRG1.get_model("compact", true),
//                     PRG2.get_model("compact", true)
//                 );
//             }
//             else {
//                 successes += 1;
//                 auto Child = opt.value();
//                 fmt::print(
//                     "Original model 1 after cross: {}\n"
//                     "Original model 2 after cross: {}\n",
//                     PRG1.get_model("compact", true),
//                     PRG2.get_model("compact", true)
//                 );
//                 fmt::print(
//                     "Crossed Model: {}\n"
//                     "=================================================\n",
//                     Child.get_model("compact", true)
//                 );
//                 Child.fit(data);
//                 auto child_pred1 = Child.predict(data);
//             }
//         }
//     }
//     ASSERT_TRUE(successes > 0);
// }

// TEST(Variation, CrossoverSizeAndDepthLimit)
// {
//     MatrixXf X(10,2);
//     ArrayXf y(10);
//     X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
//          0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,

//          0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
//          0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;

//     y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
//          3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;

//     Dataset data(X,y);

//     SearchSpace SS;
//     SS.init(data);

//     // prod operator  --> arity 4: prod(T1, T2, T3)
//     // split best     --> arity 6: if(terminal > value, T_case_true, T_case_false)
//     int max_arity = 6;

//     int successes = 0;
//     for (int d = 5; d < 15; ++d)
//     {
//         for (int s = 5; s < 15; ++s)
//         {
//             PARAMS["max_size"]  = s;
//             PARAMS["max_depth"] = d;

//             // Enforcing that the parents does not exceed max_size by
//             // taking into account the highest arity of the function nodes
//             RegressorProgram PRG1 = SS.make_regressor(d-1, s-max_arity);
//             RegressorProgram PRG2 = SS.make_regressor(d-1, s-max_arity);

//             auto PRG1_model = PRG1.get_model("compact", true);
//             auto PRG2_model = PRG2.get_model("compact", true);

//             fmt::print(
//                 "=================================================\n"
//                 "settings: depth = {}, size= {}\n"
//                 "Original model 1: {}\n"
//                 "depth = {}, size= {}\n"
//                 "Original model 2: {}\n"
//                 "depth = {}, size= {}\n",
//                 d, s, 
//                 PRG1.get_model("compact", true),
//                 PRG1.depth(), PRG1.size(),
//                 PRG2.get_model("compact", true),
//                 PRG2.depth(), PRG2.size()
//             );

//             fmt::print("cross\n");
//             auto opt = PRG1.cross(PRG2);

//             if (!opt){
//                 fmt::print("Crossover failed to create a child"
//                     "=================================================\n");
//             }
//             else {
//                 successes += 1;
//                 auto Child = opt.value();
//                 fmt::print(
//                     "Child Model      : {}\n"
//                     "Child Model depth: {}\n"
//                     "Child Model size : {}\n"
//                     "=================================================\n",
//                     Child.get_model("compact", true),
//                     Child.depth(), Child.size()
//                 );

//                 // Original didn't change
//                 ASSERT_TRUE(PRG1_model == PRG1.get_model("compact", true));
//                 ASSERT_TRUE(PRG2_model == PRG2.get_model("compact", true));

//                 // Child is within restrictions
//                 ASSERT_TRUE(Child.size() > 0);
//                 ASSERT_TRUE(Child.size() <= s);

//                 ASSERT_TRUE(Child.depth() >= 0);
//                 ASSERT_TRUE(Child.depth() <= d);
//             }
//         }
//     }
//     ASSERT_TRUE(successes > 0);
// }