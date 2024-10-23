#include "testsHeader.h"
#include "../../src/bandit/bandit.cpp"

TEST(Data, ErrorHandling)
{
    // Creating an empty dataset throws error
    EXPECT_THROW({
        MatrixXf X(0,0);
        ArrayXf y(0); 

        try
        {
            Dataset dt(X, y);
        }
        catch( const std::runtime_error& err )
        {
            const string msg = err.what();
            ASSERT_NE(
                msg.find("Error during the initialization of the dataset"),
                std::string::npos);
            throw;
        }
    }, std::runtime_error);
}

TEST(Data, MixedVariableTypes)
{
    Parameters params;

    MatrixXf X(5,3);
    X << 0  , 1,    0  , // binary with integer values
         0.0, 1.0,  1.0, // binary with float values
         2  , 1.0, -3.0, // integer with float and negative values
         2  , 1  ,  3  , // integer with integer values
         2.1, 3.7, -5.2; // float values

    X.transposeInPlace();

    ArrayXf y(3); 

    y << 6.1, 7.7, -4.2; // y = x_0 + x_1 + x_2
    
    params.functions = {
        {"Add", 0.5},
        {"Sub", 0.5},
        // a boolean operator
        {"And",       1.0},
        {"Or",        1.0},
        // operator that takes boolean as argument
        {"SplitOn",   1.0}
    };

    Dataset dt(X, y);
    SearchSpace SS;
    SS.init(dt, params.functions);

    dt.print();
    SS.print();

    for (size_t d = 5; d < 10; ++d)
        for (size_t s = 5; s < 20; ++s)
        {
            fmt::print(
                "=================================================\n"
                "depth={}, size={}. ", d, s
            );

            params.max_size  = s;
            params.max_depth = d;

            // TODO: update all calls of make_<program> to use params 
            RegressorProgram PRG = SS.make_regressor(0, 0, params);

            fmt::print(
                "Tree model: {}\n", PRG.get_model("compact", true)
            );

            // visualizing detailed information for the model
            std::for_each(PRG.Tree.begin(), PRG.Tree.end(),
                [](const auto& n) { 
                    fmt::print("Name {}, node {}, feature {}\n"
                               "  sig_hash {}\n  ret_type {}\n  ret_type type {}\n",
                               n.name, n.node_type, n.get_feature(),
                               n.sig_hash, n.ret_type, typeid(n.ret_type).name());
                });
            std::cout << std::endl;

            fmt::print( "PRG fit\n");
            PRG.fit(dt);

            fmt::print( "PRG predict\n");
            ArrayXf y_pred = PRG.predict(dt);
            fmt::print( "y_pred: {}\n", y_pred);

            // creating and fitting a child
            Variation variator = Variation<ProgramType::Regressor>(params, SS);

            Individual<PT::Regressor> IND(PRG);
            
            auto [opt, context] = variator.mutate(IND);

            if (!opt){
                fmt::print("Mutation failed to create a child\n");
            }
            else {
                auto Child = opt.value();

                fmt::print("Child program model: {}\n", Child.program.get_model("compact", true));

                fmt::print( "Child fit\n");
                Child.fit(dt);

                fmt::print( "Child predict\n");
                ArrayXf y_pred_child = Child.predict(dt);
                fmt::print( "y_pred: {}\n", y_pred_child);

                // should be the same as the fit and predict above
                fmt::print( "Child program fit\n");
                Child.program.fit(dt);

                fmt::print( "Child program predict\n");
                ArrayXf y_pred_child_program = Child.program.predict(dt);
                fmt::print( "y_pred: {}\n", y_pred_child_program);
            }
        }

    // Brush exports two DispatchTable structs named dtable_fit and dtable_predict.
    // These structures holds the mapping between nodes and its corresponding
    // operations, and are used to resolve the evaluation of an expression. 
    // dtable_fit.print();
    // dtable_predict.print();
}
TEST(Data, ShuffleTrueFalse)
{
    MatrixXf X(20,3);
    X << 0  , 1,    0  ,
         0.0, 1.0,  1.0,
         2  , 1.0, -3.0,
         2  , 1  ,  3  ,
         2.1, 3.7, -5.2,
         0  , 1,    0  ,
         0.0, 1.0,  1.0,
         2  , 1.0, -3.0,
         2  , 1  ,  3  ,
         2.1, 3.7, -5.2,
         0  , 1,    0  ,
         0.0, 1.0,  1.0,
         2  , 1.0, -3.0,
         2  , 1  ,  3  ,
         2.1, 3.7, -5.2,
         0  , 1,    0  ,
         0.0, 1.0,  1.0,
         2  , 1.0, -3.0,
         2  , 1  ,  3  ,
         2.1, 3.7, -5.2;

    X.transposeInPlace();

    ArrayXf y(20); 

    y << 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    
    // vector<string> vn = {};
    // map<string, State> Z = {};
    // vector<string> ft = {};

    // Dataset(const ArrayXXf& X, 
    //          const Ref<const ArrayXf>& y_ = ArrayXf(), 
    //          const vector<string>& vn = {}, 
    //          const map<string, State>& Z = {},
    //          const vector<string>& ft = {},
    //          bool c = false,
    //          float validation_size = 0.0,
    //          float batch_size = 1.0,
    //          bool shuffle_split = false

    Dataset dt1(X, y, {}, {}, {}, true, 0.3, 1.0, true);
    Dataset dt2(X, y, {}, {}, {}, true, 0.3, 1.0, false);
    Dataset dt3(X, y, {}, {}, {}, true, 0.0, 1.0, true);
    Dataset dt4(X, y, {}, {}, {}, true, 0.0, 1.0, false);
    Dataset dt5(X, y, {}, {}, {}, true, 1.0, 1.0, true);
    Dataset dt6(X, y, {}, {}, {}, true, 1.0, 1.0, false);

    // TODO: write some assertions here
}
