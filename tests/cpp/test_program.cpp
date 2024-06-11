#include "testsHeader.h"
#include "../../src/vary/search_space.h"
#include "../../src/program/program.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/data/io.h"

TEST(Program, MakeRegressor)
{
        
    Dataset data = Data::read_csv("docs/examples/datasets/d_enc.csv","label");

    SearchSpace SS;
    SS.init(data);
    Parameters params;

    // Program<ArrayXf> DXtree;
    for (int d = 1; d < 10; ++d)
        for (int s = 1; s < 10; ++s)
        {
            params.max_size  = s;
            params.max_depth = d;

            RegressorProgram PRG = SS.make_regressor(0, 0, params);
            fmt::print(
                "=================================================\n"
                "Tree model for depth = {}, size= {}: {}\n",
                d, s, PRG.get_model("compact", true)
            );

            auto clone = PRG.copy();
            fmt::print(
                "Copy of the original model: {}\n"
                "=================================================\n",
                clone.get_model("compact", true)
            );

            ASSERT_TRUE( PRG.get_model("compact", true)==clone.get_model("compact", true) );

            fmt::print("Models have the same representation\n");

            // weights didnt changed
            vector<float> PRG_weights(PRG.Tree.size());
            std::transform(PRG.Tree.begin(), PRG.Tree.end(), PRG_weights.begin(),
                        [&](const auto& n){ return n.get_prob_change();});

            vector<float> clone_weights(clone.Tree.size());
            std::transform(clone.Tree.begin(), clone.Tree.end(), clone_weights.begin(),
                        [&](const auto& n){ return n.get_prob_change();});
                        
            ASSERT_TRUE( PRG_weights.size()==clone_weights.size() );
            fmt::print("Models have the same number of node weights\n");

            for (size_t i=0; i<PRG_weights.size(); ++i){
                fmt::print("Weight {}: original {}, clone {}\n", i, 
                           PRG_weights.at(i), clone_weights.at(i) );
                ASSERT_TRUE( PRG_weights.at(i) == clone_weights.at(i) );
            }
            fmt::print("Models have the same node weights probabilities\n");
        }
}

TEST(Program, FitRegressor)
{
    Parameters params;

    Dataset data = Data::read_csv("docs/examples/datasets/d_enc.csv","label");

    SearchSpace SS;
    SS.init(data);

    dtable_fit.print();
    dtable_predict.print();

    // for (int t = 0; t < 10; ++t) {
        for (int d = 1; d < 10; ++d) { 
            for (int s = 1; s < 100; s+=10) {
                params.max_size  = s;
                params.max_depth = d;

                RegressorProgram PRG = SS.make_regressor(0, 0, params);
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
    Parameters params;
        
    Dataset data = Data::read_csv("docs/examples/datasets/d_enc.csv","label");

    ASSERT_FALSE(data.classification);

    SearchSpace SS;
    SS.init(data);
    
    dtable_fit.print();
    dtable_predict.print();

    // for (int t = 0; t < 10; ++t) {
        for (int d = 1; d < 10; ++d) { 
            for (int s = 1; s < 10; s+=10) {
                params.max_size  = s;
                params.max_depth = d;

                RegressorProgram PRG = SS.make_regressor(0, 0, params);
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
    Parameters params;
        
    Dataset data = Data::read_csv("docs/examples/datasets/d_analcatdata_aids.csv", "target");
    
    ASSERT_TRUE(data.classification);
    SearchSpace SS;

    SS.init(data);

    for (int d = 1; d < 10; ++d) { 
        for (int s = 1; s < 100; s+=10) {

            params.max_depth = d;
            params.max_size  = s;

            fmt::print( "Calling make_classifier...\n");
            auto PRG = SS.make_classifier(0, 0, params);

            fmt::print(
                "=================================================\n"
                "Tree model for depth = {}, size= {}: {}\n"
                "=================================================\n",
                d, s, PRG.get_model("compact", true)
            );

            fmt::print( "Fitting the model...\n");
            PRG.fit(data);
            fmt::print( "predict...\n");
            auto y = PRG.predict(data);
            fmt::print( "predict proba...\n");
            auto yproba = PRG.predict_proba(data);
        }
    }
}

TEST(Program, Serialization)
{
    Parameters params;

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
            params.max_size  = s;
            params.max_depth = d;

            RegressorProgram PRG = SS.make_regressor(0, 0, params);
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
            fmt::print( "json of initial model: {}\n", PRGjson.dump(2));

            // auto newPRG = PRGjson.get<RegressorProgram>();
            RegressorProgram newPRG = PRGjson;
            json newPRGjson = newPRG;

            fmt::print( "json of loaded model: {}\n", newPRGjson.dump(2));
            fmt::print("Initial Model: {}\n",PRG.get_model("compact", true));
            fmt::print("Loaded  Model: {}\n",newPRG.get_model("compact", true));
            
            ASSERT_TRUE(
                std::equal(PRG.Tree.begin(), PRG.Tree.end(), newPRG.Tree.begin())
            );
            newPRG.set_search_space(SS);
            newPRG.fit(data);
            ArrayXf new_y_pred = newPRG.predict(data);

        }
    }
}

TEST(Operators, ProgramSizeAndDepthPARAMS)
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

    Parameters params;

    SearchSpace SS;
    SS.init(data);

    for (int d = 1; d < 6; ++d)
    {
        for (int s = 10; s < 20; ++s)
        {
            params.max_size  = s;
            params.max_depth = d;

            fmt::print("d={},s={}\n",d,s);
            fmt::print("make_regressor\n");
            RegressorProgram PRG = SS.make_regressor(0, 0, params);
            
            fmt::print(
                "depth = {}, size= {}\n"
                "Generated Model: {}\n"
                "Model depth    : {}\n"
                "Model size     : {}\n"
                "=================================================\n",
                d, s, 
                PRG.get_model("compact", true), PRG.depth(), PRG.size()
            );

            // Terminals are weighted by default, while operators not. Since we
            // include the weights in the calculation of the size of the program,
            // and PTC2 uses the tree size (not the program size), it is not 
            // expected that initial trees will strictly respect `max_size`.
            ASSERT_TRUE(PRG.size() > 0); // size is always positive
            
            // PTC2: maximum size is s+max(arity). Since in Brush terminals are
            // weighted by default, we set it to 3*max(arity)
            ASSERT_TRUE(PRG.size() <= s+3*4); 

            ASSERT_TRUE(PRG.depth() <= d+1);
            ASSERT_TRUE(PRG.depth() > 0); // depth is always positive
        }
    }
}