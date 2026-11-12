#include "testsHeader.h"
#include <algorithm>
#include <set>
// #include "../../src/bandit/bandit.cpp"

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
    params.set_random_state(42);

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
            Variation variator = Variation<ProgramType::Regressor>(params, SS, dt);

            Individual<PT::Regressor> IND(PRG);
            
            auto opt = variator.mutate(IND);

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
    const int total = dt1.get_n_samples();

    ASSERT_TRUE(dt1.use_validation);
    ASSERT_TRUE(dt2.use_validation);
    ASSERT_EQ(dt1.get_training_data().get_n_samples() + dt1.get_validation_data().get_n_samples(), total);
    ASSERT_EQ(dt2.get_training_data().get_n_samples() + dt2.get_validation_data().get_n_samples(), total);

    ASSERT_FALSE(dt3.use_validation);
    ASSERT_FALSE(dt4.use_validation);
    ASSERT_EQ(dt3.get_training_data().get_n_samples(), total);
    ASSERT_EQ(dt3.get_validation_data().get_n_samples(), total);
    ASSERT_EQ(dt4.get_training_data().get_n_samples(), total);
    ASSERT_EQ(dt4.get_validation_data().get_n_samples(), total);

    ASSERT_FALSE(dt5.use_validation);
    ASSERT_FALSE(dt6.use_validation);
    ASSERT_EQ(dt5.get_training_data().get_n_samples(), total);
    ASSERT_EQ(dt5.get_validation_data().get_n_samples(), total);
    ASSERT_EQ(dt6.get_training_data().get_n_samples(), total);
    ASSERT_EQ(dt6.get_validation_data().get_n_samples(), total);
}

TEST(Data, ClassificationBatchesCycleTrainingRowsAndPreserveTypes)
{
    // The source has a 3:1 class ratio. With a 25% validation split, training
    // has nine class-0 and three class-1 rows. Five 25%-sized batches should
    // preserve that prevalence, retain the minority class after its unique
    // rows are exhausted, visit every training row, and never touch validation
    // rows.

    Util::r.set_seed(42);
    MatrixXf X(16, 3);
    ArrayXf y(16);
    for (int i = 0; i < 16; ++i) {
        X(i, 0) = static_cast<float>(i);      // unique row identifier (used to extract row ids later in the test)
        X(i, 1) = static_cast<float>(i % 2);  // boolean feature
        X(i, 2) = static_cast<float>(i % 4);  // categorical feature
        y(i) = i < 12 ? 0.0f : 1.0f;
    }

    Dataset data(X, y, {"id", "flag", "category"}, {},
                 {"ArrayF", "ArrayB", "ArrayI"}, true, 0.25f, 1.0f, true);
    Dataset training = data.get_training_data();
    Dataset validation = data.get_validation_data(); // we set 0.25f validation data
    training.set_batch_size(0.25f); // percentage of the training data to be used per batch (will be 0.25 of the 12 samples from the train split)

    std::set<int> training_ids;
    const auto& training_id_values = std::get<ArrayXf>(training.features.at("id"));
    for (int i = 0; i < training_id_values.size(); ++i)
        training_ids.insert(static_cast<int>(training_id_values(i)));

    std::set<int> validation_ids;
    const auto& validation_id_values = std::get<ArrayXf>(validation.features.at("id"));
    for (int i = 0; i < validation_id_values.size(); ++i)
        validation_ids.insert(static_cast<int>(validation_id_values(i)));

    std::set<int> batch_ids;

    // Each batch targets 25% of the 12 training rows, so 3 rows.
    // To include the minority class, each full batch contains 2
    // class-0 rows and 1 class-1 row.

    // There are 9 class-0 rows, so covering them takes five batches;
    // four with 2 class-0 rows and a final batch with 1. The minority class
    // is repeated in the last two batches, producing sizes 3, 3, 3, 3, 2.
    const vector<int> expected_class_zero_counts{2, 2, 2, 2, 1};
    const vector<int> expected_batch_sizes{3, 3, 3, 3, 2};

    for (int batch_number = 0; batch_number < 5; ++batch_number) {
        Dataset batch = training.get_batch();

        EXPECT_EQ(batch.feature_names, (vector<string>{"id", "flag", "category"}));
        ASSERT_EQ(batch.feature_types.size(), 3U);
        EXPECT_EQ(static_cast<int>(batch.feature_types.at(0)), static_cast<int>(DataType::ArrayF));
        EXPECT_EQ(static_cast<int>(batch.feature_types.at(1)), static_cast<int>(DataType::ArrayB));
        EXPECT_EQ(static_cast<int>(batch.feature_types.at(2)), static_cast<int>(DataType::ArrayI));
        EXPECT_EQ(batch.get_n_samples(), expected_batch_sizes.at(batch_number));

        int class_zero_count = 0;
        for (int i = 0; i < batch.y.size(); ++i)
            class_zero_count += batch.y(i) == 0.0f;

        EXPECT_EQ(class_zero_count, expected_class_zero_counts.at(batch_number));

        EXPECT_TRUE(std::any_of(batch.y.data(), batch.y.data() + batch.y.size(),
                                [](float label) { return label == 1.0f; }));

        const auto& batch_id_values = std::get<ArrayXf>(batch.features.at("id"));
        for (int i = 0; i < batch_id_values.size(); ++i)
            batch_ids.insert(static_cast<int>(batch_id_values(i)));
    }

    EXPECT_EQ(batch_ids, training_ids); // those are sets (unordered!), not lists
    EXPECT_EQ(std::count_if(training.y.data(), training.y.data() + training.y.size(),
                            [](float label) { return label == 0.0f; }), 9);

    for (int id : batch_ids)
        EXPECT_EQ(validation_ids.count(id), 0U);
}

TEST(Data, RegressionBatchesContainEveryTargetQuartile)
{
    Util::r.set_seed(42);
    MatrixXf X(16, 1);
    ArrayXf y(16);
    for (int i = 0; i < 16; ++i) {
        X(i, 0) = static_cast<float>(i);
        y(i) = static_cast<float>(i);
    }

    Dataset data(X, y, {"id"}, {}, {"ArrayF"}, false, 0.0f, 0.25f, false);
    std::set<int> seen_targets;
    for (int batch_number = 0; batch_number < 4; ++batch_number) {
        Dataset batch = data.get_batch();
        ASSERT_EQ(batch.get_n_samples(), 4);
        vector<int> quartile_counts(4, 0);
        for (int i = 0; i < batch.y.size(); ++i) {
            const int target = static_cast<int>(batch.y(i));
            ++quartile_counts.at(target / 4);
            seen_targets.insert(target);
        }
        EXPECT_EQ(quartile_counts, (vector<int>{1, 1, 1, 1}));
    }
    EXPECT_EQ(seen_targets.size(), 16U);
}
