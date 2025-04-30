#include "testsHeader.h"

// #include "../../src/simplification/constants.cpp"
// #include "../../src/simplification/inexact.cpp"

using namespace Brush::Simpl;

TEST(Simplification, ConstantsSimplification)
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

    // Constants simplification ------------------------------------------------
    RegressorProgram PRG = json(
        {{"Tree", {
            { {"node_type","Add"     },                  {"is_weighted", false} },
            { {"node_type","Constant"}, {"feature","C"}, {"is_weighted", true } },
            { {"node_type","Constant"}, {"feature","C"}, {"is_weighted", true } }

        }}, {"is_fitted_",false}}
    );

    fmt::print("PRG.fit(data);\n");
    PRG.fit(data);
    ArrayXf y_pred = PRG.predict(data);
    
    fmt::print(
        "=================================================\n"
        "Initial (fitted) Model: {}\n",
        PRG.get_model("compact", true)
    );

    Constants_simplifier constants_simplifier; 
    constants_simplifier.simplify_tree<Brush::ProgramType::Regressor>(PRG, SS, data.get_training_data());            

    fmt::print(
        "Constants-simplified Model: {}\n",
        PRG.get_model("compact", true)
    );
}

TEST(Simplification, InexactSimplification)
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
    SS.print();

    // Inexact simplification --------------------------------------------------

    // left subtree will be visited first, and it has the same predictions as the right subtree
    RegressorProgram PRG2 = json(
        {{"Tree", {
            { {"node_type","Add"     },                    {"is_weighted", false}  },

            { {"node_type","Square"  },                    {"is_weighted", true}   },
            { {"node_type","Terminal"}, {"feature","x_1"}, {"is_weighted", false } },

            { {"node_type","Mul"     },                    {"is_weighted", false} },
            { {"node_type","Constant"}, {"feature","C" },  {"is_weighted", true } },
            { {"node_type","Mul"     },                    {"is_weighted", false} },
            { {"node_type","Terminal"}, {"feature","x_1"}, {"is_weighted", false} },
            { {"node_type","Terminal"}, {"feature","x_1"}, {"is_weighted", false} }

        }}, {"is_fitted_", false}}
    );

    fmt::print("PRG2.fit(data);\n");
    PRG2.fit(data);
    ArrayXf y_pred2 = PRG2.predict(data);

    fmt::print(
        "=================================================\n"
        "Initial (fitted) Model: {}\n",
        PRG2.get_model("compact", true)
    );

    Inexact_simplifier inexact_simplifier;

    // This one requires initialization.
    // TODO: cut-off at 100 samples and use default values?
    inexact_simplifier.initUniformPlanes(16, data.get_training_data().get_n_samples(), 1);

    inexact_simplifier.simplify_tree(PRG2, SS, data.get_training_data());

    fmt::print(
        "Inexact-simplified Model: {}\n",
        PRG2.get_model("compact", true)
    );
}