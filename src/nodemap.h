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
struct NodeMap
{
    typedef std::map<std::string, NodeBase*> str_to_node;
    str_to_node node_map; 

    NodeMap(){};

    NodeMap(std::set<str_to_node> maps)
    {
        this->node_map.clear();
        for (const auto& nm : maps)
        {
            this->node_map.insert(nm.begin(), nm.end());
        }

    };

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
struct ContinuousNodeMap : NodeMap
{
    ContinuousNodeMap() 
    { 
       this->node_map = {
            { "+", new Node<T(T,T)>("+", std::plus<T>()) },
            { "-", new Node<T(T,T)>("-", std::minus<T>()) },
            { "*", new Node<T(T,T)>("*", std::multiplies<T>()) },
       };
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
    };
};

/* tuple<T,T> d_plus(T a, T b){ return {1, 1}; }; */ 
/* template<> */
/* tuple<ArrayXf,ArrayXf> d_plus(ArrayXf a, ArrayXf b) */
/* { */ 
/*     return {ArrayXf::Ones(a.size()), ArrayXf::Ones(b.size())}; */ 
/* }; */ 
/* auto d_plus[](){return tuple<T,T>{1,1}; */
template<typename T>
struct WeightedNodeMap : NodeMap
{
    WeightedNodeMap() 
    { 
       this->node_map = {
            { "+", new WeightedNode<T(T,T)>("+", 
                    std::plus<T>(), 
                    d_plus<T>()
                    )
            },
            { "-", new WeightedNode<T(T,T)>("-", 
                    std::minus<T>(),
                    d_minus<T>()
                    ) 
            },
            { "*", new WeightedNode<T(T,T)>("*", 
                    std::multiplies<T>(),
                    d_multiplies<T>()
                    ) 
            },
            { "/", new WeightedNode<T(T,T)>("/", 
                    std::divides<T>(),
                    d_divides<T>()
                    ) 
            },
            { "sin", new WeightedNode<T(T)>("sin", 
                    /* sin, */
                    [](const T& x) -> T {return sin(x);},
                    [](const T& x) -> array<T,1>{return {-cos(x)};}
                    ) 
            },
            { "cos", new WeightedNode<T(T)>("cos", 
                    [](const T& x) -> T {return cos(x);},
                    [](const T& x) -> array<T,1>{return {sin(x)};}
                    ) 
            },
            { "tanh", new WeightedNode<T(T)>("tanh", 
                    [](const T& x) -> T {return tanh(x);},
                    [](const T& x) -> array<T,1>
                        { return {1 - pow(tanh(x), 2)}; }
                    ) 
            },
            { "exp", new WeightedNode<T(T)>("exp", 
                    [](const T& x) -> T {return exp(x);},
                    [](const T& x) -> array<T,1>{return {exp(x)};}
                    ) 
            },
            { "log", new WeightedNode<T(T)>("log", 
                    safe_log<T>(),
                    d_safe_log<T>()
                    ) 
            },
            { "sqrt", new WeightedNode<T(T)>("sqrt", 
                    [](const T& x) -> T { return sqrt(abs(x)); },
                    [](const T& x) -> array<T,1> {
                        return {x/(2*sqrt(abs(x)))}; }
                    ) 
            },
            { "^2", new WeightedNode<T(T)>("^2", 
                    [](const T& x) -> T {return pow(x, 2);},
                    [](const T& x) -> array<T,1> {return {2*x}; }
                    ) 
            },
            { "^3", new WeightedNode<T(T)>("^3", 
                    [](const T& x) -> T {return pow(x, 3);},
                    [](const T& x) -> array<T,1> {return {3*pow(x, 2)}; }
                    ) 
            },
            { "^", new WeightedNode<T(T,T)>("^", 
                    [](const T& lhs, const T& rhs) -> T {return pow(lhs, rhs);},
                    [](const T& lhs, const T& rhs) -> array<T,2> {
                        return {rhs * pow(lhs, rhs-1), 
                                log(lhs) * pow(lhs, rhs)}; 
                        }
                    ) 
            },
            { "logit", new WeightedNode<T(T)>("logit", 
                    [](const T& x) -> T {return 1/(1+exp(-x));},
                    [](const T& x) -> array<T,1> {
                        return { exp(-x)/pow(1+exp(-x),2) }; }
                    ) 
            },
            { "relu", new WeightedNode<T(T)>("relu", 
                    relu<T>(),
                    d_relu<T>()
                    ) 
            },
       };
            /* { "gauss",  new Node<T(T)>(gauss, "gauss")}, */ 
            /* { "gauss2d",  new Node<T(T,T)>(gauss2d, "gauss2d")}, */ 
            /* { "logit", new Node<T(T)>(logit, "logit") }, */
            /* { "relu", new Node<T(T)>(relu, "relu") } */
    };
};
template<typename T, typename U>
struct LogicalNodeMap : NodeMap
{
    LogicalNodeMap() 
    {
       this->node_map = {
            { "<", new Node<T(U,U)>("<", lt<T,U>) },
        };
            /* { "and", new Node<T(U,U)>(Op::plus<T>, "AND") }, */
            /* { "or", new Node<T(U,U)>(Op::minus<T>, "OR") }, */
            /* { "not", new Node<T(U,U)>(Op::multiplies<T>, "NOT") }, */
            /* { "xor", new Node<T(U,U)>(Op::divides<T>, "XOR") }, */
            /* { "<=", new Node<T(U,U)>(Op::leq<U>, "LESS") }, */
            /* { "=",  new Node<T(U,U)>(Op::equal<U>, "EQUAL")}, */ 
            /* { ">",  new Node<T(U,U)>(Op::gt<U>, ">")}, */ 
            /* { ">=",  new Node<T(U,U)>(Op::geq<U>, ">")}, */ 
    };
};

ContinuousNodeMap<ArrayXf> VectorArithmeticMap; 
WeightedNodeMap<ArrayXf> DxMap; 
ContinuousNodeMap<float> FloatNodeMap; 
LogicalNodeMap<ArrayXb, ArrayXf> VectorLogicMap;
/* LogicalNodeMap<bool, float> BoolLogicMap; */
/* One node map to rule them all */
static NodeMap NM(std::set<NodeMap::str_to_node>{
                    DxMap.node_map, 
                    /* FloatNodeMap.node_map, */
                    VectorLogicMap.node_map
                    }
                 );
/* NM.node_map.insert(VectorArithmeticMap.node_map.begin(), */
/*                    VectorArithmeticMap.node_map.end()); */
/* NM.node_map.insert(VectorLogicMap.node_map.begin(), */
/*                    VectorLogicMap.node_map.end()); */

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
