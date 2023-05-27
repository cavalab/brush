#include "testsHeader.h"
#include "../../src/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"

TEST(Data, MixedVariableTypes)
{
    // We need to set at least the mutation options (and respective
    // probabilities) in order to call PRG.predict()
    PARAMS["mutation_options"] = {
        {"point",0.25}, {"insert", 0.25}, {"delete", 0.25}, {"toggle_weight", 0.25}
    };

    MatrixXf X(5,3);
    X << 0  , 1,    0  , // binary with integer values
         0.0, 1.0,  1.0, // binary with float values
         2  , 1.0, -3.0, // integer with float and negative values
         2  , 1  ,  3  , // integer with integer values
         2.1, 3.7, -5.2; // float values

    X.transposeInPlace();

    ArrayXf y(5); 

    y << 6.1, 7.7, -4.2; // y = x_0 + x_1 + x_2
    
    unordered_map<string, float> user_ops = {
        {"Add", 1},
        {"Sub", 1},
        {"SplitOn", 1}
    };

    Dataset dt(X, y);
    SearchSpace SS;
    SS.init(dt, user_ops);

    dt.print();
    SS.print();

    for (int d = 1; d < 5; ++d)
        for (int s = 1; s < 5; ++s)
        {
            
            PARAMS["max_size"]  = s;
            PARAMS["max_depth"] = d;

            RegressorProgram PRG = SS.make_regressor(d, s);
            fmt::print(
                "=================================================\n"
                "Tree model for depth = {}, size= {}: {}\n",
                d, s, PRG.get_model("compact", true)
            );

            auto Child = PRG.mutate();
            fmt::print("Child model: {}\n", Child.get_model("compact", true));

            std::for_each(PRG.Tree.begin(), PRG.Tree.end(),
                  [](const auto& n) { 
                    fmt::print("Name {}, node {}, feature {}, sig_hash {}\n",
                               n.name, n.node_type, n.get_feature(), n.sig_hash);
                   });

            std::cout << std::endl;
            
            PRG.fit(dt);
            fmt::print( "PRG predict\n");
            ArrayXf y_pred = PRG.predict(dt);
            fmt::print( "y_pred: {}\n", y_pred);

            Child.fit(dt);
            fmt::print( "Child predict\n");
            ArrayXf y_pred_child = Child.predict(dt);
            fmt::print( "y_pred: {}\n", y_pred);
        }

    // Brush exports two DispatchTable structs named dtable_fit and dtable_predict.
    // These structures holds the mapping between nodes and its corresponding
    // operations, and are used to resolve the evaluation of an expression. 
    // dtable_fit.print();
    // dtable_predict.print();
}