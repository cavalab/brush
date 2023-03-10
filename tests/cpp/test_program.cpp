#include "testsHeader.h"
#include "../../src/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"

TEST(Program, MakeRegressor)
{
        
    Dataset data = Data::read_csv("docs/examples/datasets/d_enc.csv","label");


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
        
    Dataset data = Data::read_csv("docs/examples/datasets/d_enc.csv","label");

    SearchSpace SS;
    SS.init(data);
    dtable_fit.print();
    dtable_predict.print();
    // for (int t = 0; t < 10; ++t) {
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
    // }
}

TEST(Program, PredictWithWeights)
{
        
    Dataset data = Data::read_csv("docs/examples/datasets/d_enc.csv","label");

    SearchSpace SS;
    SS.init(data);
    dtable_fit.print();
    dtable_predict.print();
    // for (int t = 0; t < 10; ++t) {
        for (int d = 1; d < 10; ++d) { 
            for (int s = 1; s < 10; s+=10) {
                RegressorProgram PRG = SS.make_regressor(d, s);
                fmt::print(
                    "=================================================\n"
                    "Tree model for depth = {}, size= {}: {}\n"
                    "=================================================\n",
                    d, s, PRG.get_model("compact", true)
                );
                PRG.fit(data);
                auto y = PRG.predict(data);
                auto weights = PRG.get_weights();
                auto yweights = PRG.predict_with_weights(data, weights);
                for (int i = 0; i < y.size(); ++i){
                    if (std::isnan(y(i)))
                        ASSERT_TRUE(std::isnan(y(i)));
                    else
                        ASSERT_FLOAT_EQ(y(i), yweights(i));
                }
            }
        }
    // }
}

TEST(Program, FitClassifier)
{
        
    Dataset data = Data::read_csv("docs/examples/datasets/d_analcatdata_aids.csv","target");
    SearchSpace SS;
    SS.init(data);

    // for (int t = 0; t < 10; ++t) {
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
    // }
}



TEST(Program, Mutation)
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

TEST(Program, Serialization)
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
            fmt::print(
                "=================================================\n"
                "depth = {}, size= {}\n"
                "Initial Model: {}\n",
                d, s, 
                PRG.get_model("compact", true)
            );
            PRG.fit(data);
            ArrayXf y_pred = PRG.predict(data);
            json PRGjson = PRG;
            fmt::print( "json: {}\n", PRGjson.dump(2));
            auto newPRG = PRGjson.get<RegressorProgram>();
            fmt::print("Initial Model: {}\n",PRG.get_model("compact", true));
            fmt::print("Loaded  Model: {}\n",newPRG.get_model("compact", true));
            ASSERT_TRUE(
                std::equal(PRG.Tree.begin(),PRG.Tree.end(),newPRG.Tree.begin())
            );
            newPRG.set_search_space(SS);
            newPRG.fit(data);
            ArrayXf new_y_pred = newPRG.predict(data);


        }
    }
}

TEST(Program, OptimizeAdditionPositiveWeights)
{
    /* @brief Tests whether weight optimization works on a simple additive problem.
        The dataset models y = 2*x1 + 3*x2. 
        The initial model is yhat = 1*x1 + 1*x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_plus_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Add"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();
    ArrayXf true_weights(2);
    true_weights << 2.0, 3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeAdditionNegativeWeights)
{
    /* @brief Tests whether weight optimization works on a simple additive problem.
        The dataset models y = 2*x1 - 3*x2. 
        The initial model is yhat = 1*x1 + 1*x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_subtract_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Add"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();
    ArrayXf true_weights(2);
    true_weights << 2.0, -3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
}

TEST(Program, OptimizeSubtractionPositiveWeights)
{
    /* @brief Tests whether weight optimization works on a simple additive problem.
        The dataset models y = 2*x1 - 3*x2. 
        The initial model is yhat = 1*x1 - 1*x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_subtract_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Sub"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();
    ArrayXf true_weights(2);
    true_weights << 2.0, 3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 

TEST(Program, OptimizeSubtractionNegativeWeights)
{
    /* @brief Tests whether weight optimization works on a simple additive problem.
        The dataset models y = 2*x1 + 3*x2. 
        The initial model is yhat = 1*x1 - 1*x2. 
        The test checks that the target output and initial model are close, and also
        whether the weights are correct. Given that the model yhat has the correct
        structure, the fitted model should have an infinitesimally small error.
    */
    Dataset data = Data::read_csv("docs/examples/datasets/d_2x1_plus_3x2.csv","target");
    SearchSpace SS;
    SS.init(data);

    json PRGjson = {
        {"Tree", {
        {
            {"node_type","Sub"},
            {"is_weighted", true}
        },
        {
            {"node_type","Terminal"},
            {"feature","x1"},
        },
        {
            {"node_type","Terminal"},
            {"feature","x2"},
        }
        }},
        {"is_fitted_",false}
    };
    fmt::print( "initial json: {}\n", PRGjson.dump(2));

    // make program from json
    RegressorProgram PRG = PRGjson;

    // make json from the program just to visually check 
    json loadedPRGjson = PRG;
    fmt::print( "loaded json: {}\n", loadedPRGjson.dump(2));

    // fit model
    fmt::print( "fit\n");
    PRG.fit(data);
    fmt::print( "predict\n");
    ArrayXf y_pred = PRG.predict(data);

    auto learned_weights = PRG.get_weights();
    ArrayXf true_weights(2);
    true_weights << 2.0, -3.0;

    // calculating the MSE
    float mse = (data.y - y_pred).square().mean();

    ASSERT_TRUE(data.y.isApprox(y_pred, 1e-4));
    ASSERT_TRUE(learned_weights.isApprox(true_weights, 1e-4));
    ASSERT_TRUE(mse <= 1e-4);
} 