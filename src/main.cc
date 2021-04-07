#include <algorithm>
#include <iostream>
#include "node.h"
#include "operators.h"
#include "program.h"
#include "data.h"
#include "tree.h"

using namespace Eigen;
using namespace std;
using namespace Brush;
using namespace Brush::Dat;

/* template<typename T> */
/* T add(T x, T y){ return x + y; }; */
/* std::map<std::string, NodeBase* > NodeMap = { */
/*     { "+", new Node<ArrayXf(ArrayXf,ArrayXf)>(Op::plus<ArrayXf>, "ADD") }, */
/*     { "-", new Node<ArrayXf(ArrayXf,ArrayXf)>(Op::minus<ArrayXf>, "MINUS") }, */
/*     { "<", new Node<ArrayXb(ArrayXf, ArrayXf)>(Op::lt<ArrayXf>, "LESS THAN") }, */
/*     { "*", new Node<ArrayXf(ArrayXf,ArrayXf)>(Op::multiplies<ArrayXf>, "TIMES") }, */
/* }; */

int main(int, char **)
{
    // auto float_binary_operators = make_binary_operators<float>();
    // auto array_binary_operators = make_binary_operators<ArrayXf>();
    // vector<BinaryOperator<float>*> FloatBinaryOperators = { 
    //                             new Add<float>(),
    //                             new Sub<float>()
    //                             };
    // vector<Operator<Eigen::ArrayXf>*> ArrayBinaryOperators = { 
    //                             new Add<float>(),
    //                             new Sub<float>()
    //                             };
    // auto op = Add<float>();
    BinaryOperator<float>* op = new Add<float>();
    auto node = WeightedDxNode<float(float,float)>(
        op->get_name(),
        op->f,
        op->df
    );
    // for (const auto& op : float_binary_operators)
    // for (const auto& op : FloatBinaryOperators)
    // {
    //     auto node = WeightedDxNode<float(const float&,const float&)>(
    //                                op->get_name(), 
    //                                (*op), 
    //                                op->df
    //                                );
    // }
    // for (const auto& op : array_binary_operators)
    // {
    //     auto node = WeightedDxNode<ArrayXf(ArrayXf, ArrayXf)>(
    //                                op->get_name(), 
    //                                op, 
    //                                op->df
    //                                );
    // }
    // cout << "declaring node...\n";
    // Node<int(int, int)> node_plus("+", std::plus<int>());
    // cout << "3+5 = " << node_plus.op(3, 5) << endl;
    // cout << "3.1+5.7 = " << node_plus.op(3.1, 5.7) << endl;

    // Node<float(float, float)> node_plus_float("+", std::plus<float>());

    // cout << "3+5 = " << node_plus_float.op(3, 5) << endl;
    // cout << "3.1+5.7 = " << node_plus_float.op(3.1, 5.7) << endl;

    // Node<bool(float, float)> node_lt("<", std::less<float>() );
    // cout << "3<5 = " << node_lt.op(3, 5) << endl;
    // cout << "3.1<5.7 = " << node_lt.op(3.1, 5.7) << endl;

    // Node<float(float, float)> node_minus("-", std::minus<float>());
    // cout << "3-5 = " << node_minus.op(3, 5) << endl;
    // cout << "3.1-5.7 = " << node_minus.op(3.1, 5.7) << endl;

    // Node<float(float, float)> node_multiplies("*", std::multiplies<float>());
    // cout << "3*5 = " << node_multiplies.op(3, 5) << endl;
    // cout << "3.1*5.7 = " << node_multiplies.op(3.1, 5.7) << endl;
    // cout << "dune.\n";

    // // construct tree
    // // P = LESS ( ADD (X1, X2) , MULTIPLY( X1, X2) )
    // //
    // Program<ArrayXb> tree;
    // auto top = tree.prg.begin();
    // auto root = tree.prg.insert(tree.prg.begin(), NM["<"]);
    // auto add = tree.prg.append_child(root, NM["+"]);
    // /* auto add = tree.append_child(root, new Node<ArrayXf(ArrayXf,ArrayXf)>( */
    // /*             std::plus<ArrayXf>(), "ADD")); */
    
    // auto times = tree.prg.append_child(root, NM["*"]);
    // tree.prg.append_child(add, new Node<ArrayXf>("x_1", 0));
    // tree.prg.append_child(add, new Node<ArrayXf>("x_2", 1));
    // tree.prg.append_child(times, new Node<ArrayXf>("x_1", 0));
    // tree.prg.append_child(times, new Node<ArrayXf>("x_2", 1));

    // auto loc = tree.prg.begin(); 

    // while(loc!=tree.prg.end()) 
    // {
    //     for(int i=0; i<tree.prg.depth(loc); ++i)
    //         cout << "--";
    //     cout << (*loc)->name << endl;
    //     ++loc;
    //     /* cout << endl; */
    // }

    // cout << "=======================================\n";
    // cout << "setting up data...\n";
    // MatrixXf X(2,10);
    // ArrayXf y(10);
    // X << 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
    //      2.0,1.0,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0,
    // y << 1.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0;
    // Longitudinal Z;
    // Data d(X,y,Z);
    // // cout << "fitting tree...\n";
    // // State out = tree.fit(d);
    // // cout << "output: " << get<ArrayXb>(out) << endl;

    // // test backpropagation
    // // y = 3.14*X1 + 1.68*X2
    // // r: random numbers
    // // X1: cos(r)
    // // X2: sin(r)
    // /* r = 0.54340523, 0.98342536, 0.52725502, 0.1019157 , 0.13250716, */
    // /*          0.2273736 , 0.78280196, 0.31946665, 0.07088554, 0.99791986; */
    // X << 0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376,
    //      0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151,
    //      0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973,
    //      0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526;
    // y << 3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
    //          3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179;

            
    // Program<ArrayXf> DXtree;
    // cout << "making program...\n";
    // for (int d = 1; d < 10; ++d)
    //     for (int s = 1; s < 50; ++s)
    //         DXtree.make_program(d, s);
    // // auto root2 = DXtree.prg.insert(DXtree.prg.begin(), NM["+"]);
    // // DXtree.prg.append_child(root2, new Node<ArrayXf>("x_1", 0));
    // // DXtree.prg.append_child(root2, new Node<ArrayXf>("x_2", 1));
    // DXtree.fit(d);
    // cout << "generating predictions\n";
    // ArrayXf y_pred = DXtree.predict(d);
    // cout << "gradient descent\n";
    // cout << "calculating loss\n";
    // cout << "y_pred: " << y_pred.transpose() << endl;
    // cout << "y: " << y.transpose() << endl;
    // cout << "loss: " << (y_pred - y).square().transpose() << endl;
    // ArrayXf d_loss = 2*(y_pred - y);
    // for (int i = 0; i < 20; ++i)
    // {
    //     DXtree.grad_descent(d_loss, d);
    //     y_pred = DXtree.predict(d);
    //     cout << "updated y_pred: " << y_pred.transpose() << endl;
    //     cout << "             y: " << y.transpose() << endl;
    //     cout << "loss: " << (y_pred - y).square().transpose() << endl;
    //     d_loss = 2*(y_pred - y);
    // }


}
