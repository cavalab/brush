#include <vector>
using namespace std;

TEST(Program, UpdateWeights)
{
        
    Dataset data = Data::read_csv("examples/datasets/d_enc.csv","label");

    SearchSpace SS;
    SS.init(data);

    for (int d = 1; d < 10; ++d) { 
        for (int s = 1; s < 10; s+=10) {
            RegressorProgram PRG = SS.make_regressor(d, s);
            fmt::print(
                "=================================================\n"
                "Tree model for depth = {}, size= {}: {}\n"
                "=================================================\n",
                d, s, PRG.get_model("compact", true)
            );

            auto weights = PRG.get_weights();
            auto n = weights.size;
            auto new_weights = vector<float>(n, 1.3);
            PRG.set_weights(new_weights);
            auto weights = PRG.get_weights();
            
            for (int i = 0; i < weights.size(); ++i){
                if (std::isnan(weights(i)))
                    ASSERT_TRUE(std::isnan(weights(i)));
                else
                    ASSERT_FLOAT_EQ(1.3, weights(i));
            }
        }
    }
}