#include "testsHeader.h"

#include "../../src/bandit/bandit.h"
#include "../../src/bandit/bandit_operator.h"
#include "../../src/bandit/dummy.h"
#include "../../src/bandit/thompson.h"
#include "../../src/bandit/linear_thompson.h"

#include "../../src/bandit/bandit.cpp"
#include "../../src/bandit/bandit_operator.cpp"
#include "../../src/bandit/dummy.cpp"
#include "../../src/bandit/thompson.cpp"
#include "../../src/bandit/linear_thompson.cpp"

using namespace Brush::MAB;
using testing::TestWithParam;

class BanditTest 
    : public TestWithParam< std::tuple<string, std::map<string, float>> > {
    protected:
        void SetUp() override {
            // Unpack test settings into the variables
            std::tie(banditType, expectedProbs) = GetParam();
        }
        // void TearDown() override { }

        // Those parameters will be accessible inside TEST_P(OptimizerTest, ...)
        string banditType;
        std::map<string, float> expectedProbs;
};

TEST_P(BanditTest, BanditProbabilities) {
    // Create a DummyBandit with two arms
    std::vector<std::string> arms = {"foo1", "foo2"};
    Bandit<string> bandit(banditType, arms);

    // this is required in order for it to work
    bandit.set_bandit();

    std::cout << "Bandit type: " << bandit.get_type() << std::endl;
    std::cout << "Bandit arms: ";
    for (const auto& arm : bandit.get_arms()) {
        std::cout << arm << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(bandit.get_type(), banditType);
    EXPECT_EQ(bandit.get_arms(), arms);

    // Print the size of the bandit probabilities
    auto curr_probs = bandit.sample_probs();
    std::cout << "Bandit probabilities size: " << curr_probs.size() << std::endl;
    
    std::cout << "Bandit probabilities: " << endl;
    for (const auto& prob : curr_probs) {
        std::cout << "{" << prob.first << ", " << prob.second << "} " << endl;
    }

    // Check if the bandit initial probabilities are set correctly
    std::map<string, float> initialProbs = {{"foo1", 0.5}, {"foo2", 0.5}};
    EXPECT_EQ(bandit.sample_probs(false), initialProbs);
    
    // things dont change
    std::map<string, float> sampledProbs = bandit.sample_probs(true);

    // Context we need to pass to the bandit    
    VectorXf context(3);
    context << 1, 1, 1;
    
    // Update the bandit with arm 1 and reward 1.0
    bandit.choose(context); // linear thompson needs to choose before any update
    bandit.update("foo1", 1.0, context);
    
    // Sample the bandit probabilities after updating
    sampledProbs = bandit.sample_probs(true);
    
    // Check if the updated probabilities are equal to the expected probabilities
    // EXPECT_EQ(sampledProbs, expectedProbs);

    std::cout << std::endl;
    std::cout << "Sampled probabilities: ";
    for (const auto& prob : sampledProbs) {
        std::cout << "{" << prob.first << ", " << prob.second << "} ";
    }
    std::cout << std::endl;
}

INSTANTIATE_TEST_SUITE_P(BanditTypes, BanditTest, testing::Values(
    std::make_tuple("dummy", std::map<string, float>{{"foo1", 0.5}, {"foo2", 0.5}}),
    std::make_tuple("thompson", std::map<string, float>{{"foo1", 0.666667}, {"foo2", 0.50000}}),
    std::make_tuple("dynamic_thompson", std::map<string, float>{{"foo1", 0.5}, {"foo2", 0.5}}),
    std::make_tuple("linear_thompson", std::map<string, float>{{"foo1", 0.5}, {"foo2", 0.5}})
));

TEST(BanditTest, GetContext) {
    // // Create a DummyBandit with two arms
    // std::vector<std::string> arms = {"foo1", "foo2"};
    // Bandit<string> bandit("linear_thompson", arms, 1);

    // // this is required in order for it to work
    // bandit.set_bandit();

    // // Create a dummy tree and search space
    // tree<Node> dummyTree;
    // auto root = dummyTree.insert(dummyTree.begin(), Node{"root", NodeType::Operator});
    // auto child1 = dummyTree.append_child(root, Node{"child1", NodeType::Terminal});
    // auto child2 = dummyTree.append_child(root, Node{"child2", NodeType::Terminal});

    // SearchSpace ss;
    // ss.terminal_map = {{"type1", {Node{"child1", NodeType::Terminal}, Node{"child2", NodeType::Terminal}}}};
    // ss.terminal_types = {"type1"};

    // // Get context for a specific spot in the tree
    // VectorXf context = bandit.get_context<T>(dummyTree, child1, ss);

    // std::cout << "Context: " << context.transpose() << std::endl;

    // // Check if the context size is correct
    // EXPECT_EQ(context.size(), 3 * (NodeTypes::Count + ss.terminal_map["type1"].size()));

    // // Check if the context values are as expected
    // // This is a simple check, you might need to adjust it based on your actual context calculation
    // EXPECT_EQ(context.sum(), 1.0);
}
