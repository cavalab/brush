/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODELIST_H
#define NODELIST_H
//internal includes
#include "node.h"
#include "operators.h"

/* template<typename R, typename Arg1, typename Arg2=Arg1> */
template<typename T, typename ... Args>
struct NodeMap
{
    std::map<std::string, NodeBase*> node_map; 

    ~NodeMap()
    {
		for (auto it = node_map.cbegin(); it != node_map.cend(); )
		{
            node_map.erase(it++);    // or "it = m.erase(it)" since C++11
        }
    };

    NodeBase* operator[](const std::string& op)
    { 
        if (node_map.find(op) == node_map.end())
            std::cerr << "ERROR: couldn't find " << op << endl;
        
        return this->node_map.at(op); 
    };
};

template<typename T>
struct ContinuousNodeMap : NodeMap<T>
{
    ContinuousNodeMap() 
    { 
       this->node_map = std::map<std::string, NodeBase*>({
            { "+", new Node<T(T,T)>(std::plus<T>(), "ADD") },
            { "-", new Node<T(T,T)>(std::minus<T>(), "SUBTRACT") },
            { "*", new Node<T(T,T)>(std::multiplies<T>(), "TIMES") }
            /* { "/", Node<T(T,T)>(Op::safe_divide<T>, "DIV") }, */
            /* { "sqrt",  new Node<T(T)>(sqrt, "sqrt")}, */ 
            /* { "sin",  new Node<T(T)>(sin, "sin")}, */ 
            /* { "cos",  new Node<T(T)>(cos, "cos")}, */ 
            /* { "tanh",  new Node<T(T)>(tanh, "tanh")}, */ 
            /* { "^2",  new Node<T(T)>(square, "^2")}, */ 
            /* { "^3",  new Node<T(T)>(cube, "^3")}, */ 
            /* { "^",  new Node<T(T)>(^, "^")}, */ 
            /* { "exp",  new Node<T(T)>(exp, "exp")}, */ 
            /* { "gauss",  new Node<T(T)>(gauss, "gauss")}, */ 
            /* { "gauss2d",  new Node<T(T,T)>(gauss2d, "gauss2d")}, */ 
            /* { "log", new Node<T(T)>(log, "log") }, */   
            /* { "logit", new Node<T(T)>(logit, "logit") }, */
            /* { "relu", new Node<T(T)>(relu, "relu") } */
       });
    };
};

template<typename T, typename U>
struct LogicalNodeMap : NodeMap<T,U>
{
    LogicalNodeMap() 
    {
       this->node_map = std::map<std::string, NodeBase*>(
        {
            /* { "and", new Node<T(U,U)>(Op::plus<T>, "AND") }, */
            /* { "or", new Node<T(U,U)>(Op::minus<T>, "OR") }, */
            /* { "not", new Node<T(U,U)>(Op::multiplies<T>, "NOT") }, */
            /* { "xor", new Node<T(U,U)>(Op::divides<T>, "XOR") }, */
            { "<", new Node<T(U,U)>(Op::lt<T,U>, "LESS") }
            /* { "<=", new Node<T(U,U)>(Op::leq<U>, "LESS") }, */
            /* { "=",  new Node<T(U,U)>(Op::equal<U>, "EQUAL")}, */ 
            /* { ">",  new Node<T(U,U)>(Op::gt<U>, ">")}, */ 
            /* { ">=",  new Node<T(U,U)>(Op::geq<U>, ">")}, */ 
        }
        );
    };
};

ContinuousNodeMap<ArrayXf> VectorArithmeticMap; 
ContinuousNodeMap<float> FloatNodeMap; 
LogicalNodeMap<ArrayXb, ArrayXf> VectorLogicMap;
/* LogicalNodeMap<bool, float> BoolLogicMap; */

#endif

// Old nodes from FEAT:
    /* { "+",  new NodeAdd({1.0,1.0})}, */ 
    /* { "-",  new NodeSubtract({1.0,1.0})}, */ 
    /* { "*",  new NodeMultiply({1.0,1.0})}, */ 
    /* { "/",  new NodeDivide({1.0,1.0})}, */ 
    /* { "sqrt",  new NodeSqrt({1.0})}, */ 
    /* { "sin",  new NodeSin({1.0})}, */ 
    /* { "cos",  new NodeCos({1.0})}, */ 
    /* { "tanh",  new NodeTanh({1.0})}, */ 
    /* { "^2",  new NodeSquare({1.0})}, */ 
    /* { "^3",  new NodeCube({1.0})}, */ 
    /* { "^",  new NodeExponent({1.0})}, */ 
    /* { "exp",  new NodeExponential({1.0})}, */ 
    /* { "gauss",  new NodeGaussian({1.0})}, */ 
    /* { "gauss2d",  new Node2dGaussian({1.0, 1.0})}, */ 
    /* { "log", new NodeLog({1.0}) }, */   
    /* { "logit", new NodeLogit({1.0}) }, */
    /* { "relu", new NodeRelu({1.0}) }, */
    /* { "b2f", new NodeFloat<bool>() }, */
    /* { "c2f", new NodeFloat<int>() }, */
    
    /* // logical operators */
    /* { "and", new NodeAnd() }, */
    /* { "or", new NodeOr() }, */
    /* { "not", new NodeNot() }, */
    /* { "xor", new NodeXor() }, */
    /* { "=", new NodeEqual() }, */
    /* { ">", new NodeGreaterThan() }, */
    /* { ">=", new NodeGEQ() }, */        
    /* { "<", new NodeLessThan() }, */
    /* { "<=", new NodeLEQ() }, */
    /* { "split", new NodeSplit<float>() }, */
    /* { "fuzzy_split", new NodeFuzzySplit<float>() }, */
    /* { "fuzzy_fixed_split", new NodeFuzzyFixedSplit<float>() }, */
    /* { "split_c", new NodeSplit<int>() }, */
    /* { "fuzzy_split_c", new NodeFuzzySplit<int>() }, */
    /* { "fuzzy_fixed_split_c", new NodeFuzzyFixedSplit<int>() }, */
    /* { "if", new NodeIf() }, */   	    		
    /* { "ite", new NodeIfThenElse() }, */
    /* { "step", new NodeStep() }, */
    /* { "sign", new NodeSign() }, */
       
    /* // longitudinal nodes */
    /* { "mean", new NodeMean() }, */
    /* { "median", new NodeMedian() }, */
    /* { "max", new NodeMax() }, */
    /* { "min", new NodeMin() }, */
    /* { "variance", new NodeVar() }, */
    /* { "skew", new NodeSkew() }, */
    /* { "kurtosis", new NodeKurtosis() }, */
    /* { "slope", new NodeSlope() }, */
    /* { "count", new NodeCount() }, */
    /* { "recent", new NodeRecent() }, */
