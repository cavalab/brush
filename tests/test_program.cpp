#include "testsHeader.h"
#include "../src/search_space.h"
#include "../src/program.h"
#include "../src/operator.h"

TEST(Program, MakeProgram)
{
        
    ArrayXXf X(10,2);
    ArrayXf y(10);
    X << 1.1,2.0,3.0,4.0,5.0,6.5,7.0,8.0,9.0,10.0,
         2.0,1.2,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0,
    y << 1.0,0.0,1.4,1.0,0.0,1.0,1.0,0.0,0.0,0.0;
    Data data(X,y);
    for (const auto& kv : data.features)
    {
        const string& name = kv.first;
        const State& value = kv.second;

        std::cout << name << ": " << name;
        /* std::visit([](auto&& v){std::cout << v << endl;}, value); */
    }
    // State out = tree.fit(d);
    // cout << "output: " << get<ArrayXb>(out) << endl;

    map<string, float> user_ops = {
        {"Add", 1},
        {"Sub", 1},
        {"Div", .5},
        {"Times", 0.5}
    };

    SearchSpace SS;
    SS.init(data,user_ops);

            
    // Program<ArrayXf> DXtree;
    for (int d = 1; d < 10; ++d)
        for (int s = 1; s < 50; ++s)
        {
            Program<ArrayXf> PRG(SS, d, 0, s);
            cout << "=================================================" << "\n";
            cout << "Tree model for depth = " << d << ", size = " << s << ":\n";
            /* cout << PRG.get_tree_model(true) << endl; */
            cout << PRG.get_model(true) << endl;
            cout << "=================================================" << "\n";
        }
}

TEST(Program, BackProp)
{
    // test backpropagation
    // y = 3.14*X1 + 1.68*X2
    // r: random numbers
    // X1: cos(r)
    // X2: sin(r)
    /* r = 0.54340523, 0.98342536, 0.52725502, 0.1019157 , 0.13250716, */
    /*          0.2273736 , 0.78280196, 0.31946665, 0.07088554, 0.99791986; */
    MatrixXf X(10,2);
    ArrayXf y(10);
    X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
         0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,
         0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
         0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;
    y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
             3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;
    Data data(X,y);

    SearchSpace SS;
    SS.init(data);

    Program<ArrayXf> DXtree(SS);
    // auto root = DXtree.prg.insert(DXtree.prg.begin(), SS.get_op(typeid(ArrayXf));
    // DXtree.prg.append_child(root2, new Node<ArrayXf>("x_1", 0));
    // DXtree.prg.append_child(root2, new Node<ArrayXf>("x_2", 1));
    DXtree.fit(data);
    cout << "generating predictions\n";
    ArrayXf y_pred = DXtree.predict(data);
    cout << "gradient descent\n";
    cout << "calculating loss\n";
    cout << "y_pred: " << y_pred.transpose() << endl;
    cout << "y: " << y.transpose() << endl;
    cout << "loss: " << (y_pred - y).square().transpose() << endl;
    ArrayXf d_loss = 2*(y_pred - y);
    for (int i = 0; i < 20; ++i)
    {
        DXtree.grad_descent(d_loss, data);
        y_pred = DXtree.predict(data);
        cout << "updated y_pred: " << y_pred.transpose() << endl;
        cout << "             y: " << y.transpose() << endl;
        cout << "loss: " << (y_pred - y).square().transpose() << endl;
        d_loss = 2*(y_pred - y);
    }

}

TEST(Program, Mutation)
{
    // test mutation
    // y = 3.14*X1 + 1.68*X2
    // r: random numbers
    // X1: cos(r)
    // X2: sin(r)
    /* r = 0.54340523, 0.98342536, 0.52725502, 0.1019157 , 0.13250716, */
    /*          0.2273736 , 0.78280196, 0.31946665, 0.07088554, 0.99791986; */
    // TODO: set random seed
    MatrixXf X(10,2);
    ArrayXf y(10);
    X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
         0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,
         0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
         0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;
    y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
             3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;
    Data data(X,y);

    SearchSpace SS;
    SS.init(data);

    Program<ArrayXf> DXtree(SS, 6, 10, 30);
    cout << "starting model:\n";
    cout << DXtree.get_tree_model(true) << endl;
    DXtree.fit(data);
    cout << "generating predictions\n";
    ArrayXf y_pred = DXtree.predict(data);
    cout << "y_pred: " << y_pred.transpose() << endl;
    cout << "mutating...\n";
    auto Child = DXtree.mutate();

    cout << "=================================================" << "\n";
    cout << "Mutated model:" << endl ;
    cout << Child.get_tree_model(true) << endl;
    cout << "=================================================" << "\n";

    Child.fit(data);
    cout << "generating predictions\n";
    y_pred = Child.predict(data);
    cout << "y_pred: " << y_pred.transpose() << endl;

}
