// #include "testsHeader.h"
// #include "../../src/program/program.h"
// #include "../../src/search_space.h"
// #include "../../src/program/dispatch_table.h"

// TEST(Data, ErrorHandling)
// {
//     // Creating an empty dataset throws error
//     EXPECT_THROW({
//         MatrixXf X(0,0);
//         ArrayXf y(0); 

//         try
//         {
//             Dataset dt(X, y);
//         }
//         catch( const std::runtime_error& err )
//         {
//             const string msg = err.what();
//             ASSERT_NE(
//                 msg.find("Error during the initialization of the dataset"),
//                 std::string::npos);
//             throw;
//         }
//     }, std::runtime_error);
// }

// TEST(Data, MixedVariableTypes)
// {
//     // We need to set at least the mutation options (and respective
//     // probabilities) in order to call PRG.predict()
//     PARAMS["write_mutation_trace"] = true;
//     PARAMS["mutation_options"] = {
//         {"point",0.167}, {"insert", 0.167}, {"delete", 0.167}, {"subtree", 0.167}, {"toggle_weight_on", 0.167}, {"toggle_weight_off", 0.167}
//     };

//     MatrixXf X(5,3);
//     X << 0  , 1,    0  , // binary with integer values
//          0.0, 1.0,  1.0, // binary with float values
//          2  , 1.0, -3.0, // integer with float and negative values
//          2  , 1  ,  3  , // integer with integer values
//          2.1, 3.7, -5.2; // float values

//     X.transposeInPlace();

//     ArrayXf y(3); 

//     y << 6.1, 7.7, -4.2; // y = x_0 + x_1 + x_2
    
//     unordered_map<string, float> user_ops = {
//         {"Add", 0.5},
//         {"Sub", 0.5},
//         // a boolean operator
//         {"And",       1.0},
//         {"Or",        1.0},
//         // operator that takes boolean as argument
//         {"SplitOn",   1.0}
//     };

//     Dataset dt(X, y);
//     SearchSpace SS;
//     SS.init(dt, user_ops);

//     dt.print();
//     SS.print();

//     for (size_t d = 5; d < 10; ++d)
//         for (size_t s = 5; s < 20; ++s)
//         {
//             fmt::print(
//                 "=================================================\n"
//                 "depth={}, size={}. ", d, s
//             );

//             PARAMS["max_size"]  = s;
//             PARAMS["max_depth"] = d;

//             RegressorProgram PRG = SS.make_regressor(s-4, d-4);

//             fmt::print(
//                 "Tree model: {}\n", PRG.get_model("compact", true)
//             );

//             // visualizing detailed information for the model
//             std::for_each(PRG.Tree.begin(), PRG.Tree.end(),
//                 [](const auto& n) { 
//                     fmt::print("Name {}, node {}, feature {}\n"
//                                "  sig_hash {}\n  ret_type {}\n  ret_type type {}\n",
//                                n.name, n.node_type, n.get_feature(),
//                                n.sig_hash, n.ret_type, typeid(n.ret_type).name());
//                 });

//             std::cout << std::endl;

//             fmt::print( "PRG fit\n");
//             PRG.fit(dt);
//             fmt::print( "PRG predict\n");
//             ArrayXf y_pred = PRG.predict(dt);
//             fmt::print( "y_pred: {}\n", y_pred);

//             // creating and fitting a child
//             auto opt = PRG.mutate();

//             if (!opt){
//                 fmt::print("Mutation failed to create a child\n");
//                 fmt::print("{}\n", PARAMS["mutation_trace"].get<json>().dump());
//             }
//             else {
//                 auto Child = opt.value();

//                 fmt::print("Child model: {}\n", Child.get_model("compact", true));

//                 fmt::print( "Child fit\n");
//                 Child.fit(dt);
//                 fmt::print( "Child predict\n");
//                 ArrayXf y_pred_child = Child.predict(dt);
//                 fmt::print( "y_pred: {}\n", y_pred);
//             }
//         }

//     // Brush exports two DispatchTable structs named dtable_fit and dtable_predict.
//     // These structures holds the mapping between nodes and its corresponding
//     // operations, and are used to resolve the evaluation of an expression. 
//     // dtable_fit.print();
//     // dtable_predict.print();
// }