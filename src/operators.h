/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef OP_H
#define OP_H
#include <Eigen/Dense>
#include "init.h"
#include "util/utils.h"
using namespace Brush::Util;

using namespace std;
// using namespace Brush;

typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using Eigen::ArrayBase;
using Eigen::Array;
using Eigen::ArrayXi;


namespace Brush
{
/* Partial Derivative Functions 
 *
 * These fns take partial derivates of fns f(x1, ... xm).
 * Each returns an array of partial derivatives of the form 
 * df/dx1, ..., df/dxm 
 */

/// add
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, array<T,2>> 
d_add(const T &lhs, const T &rhs) 
{
    return {1, 1};
};

/// add specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, array<T,2>> 
d_add(const T &lhs, const T &rhs) 
{
    return {T::Ones(lhs.size()), T::Ones(rhs.size())}; 
};

/// sub
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, array<T,2>> 
d_sub(const T &lhs, const T &rhs)  
{
    return {1, -1};
};
/// sub specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, array<T,2>> 
d_sub(const T &lhs, const T &rhs)  
{
    return {T::Ones(lhs.size()), -T::Ones(rhs.size())}; 
};

/// multiply
template<typename T>
array<T,2> d_times(const T &lhs, const T &rhs)  
{
    return {rhs, lhs};
}

/// divide
template<typename T>
array<T,2> d_div(const T &lhs, const T &rhs) 
{
    return {1/rhs, -lhs/(pow(rhs,2))};
};

/// log
template<typename T>
array<T,1> d_safe_log(const T &x)
{
    return {Brush::Util::limited(T(1/x))};
};

/// relu
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, array<T,1>> 
d_relu(const T &x) 
{
    return {x > 0 ? 1 : 0.01};
};

/// relu specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, array<T,1>> 
d_relu(const T &x)  
{
    return {(x > 0).select(ArrayXf::Ones(x.size()), 
                            ArrayXf::Zero(x.size())+0.01)};
}

/// logit
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, array<T,1>> 
d_logit(const T &x) 
{
    return { exp(-x)/pow(1+exp(-x), float(2)) }; 
};

/// logit specialization for Eigen Arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, array<T,1>> 
d_logit(const T &x)  
{
    return { exp(-x)/(1+exp(-x)).pow(2) };
};

template<typename T>
array<T,2> d_pow(const T& lhs, const T& rhs) 
{
    return {rhs * pow(lhs, rhs-1), log(lhs) * pow(lhs, rhs)}; 
};

////////////////////////////////////////////////////////////////////////////////

/*! Operator definitions */

/// safe log
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
safe_log(const T& x)
{
    if (fabs(x) > NEAR_ZERO)
        return log(T(abs(x)));
    else
        return MIN_FLT;
};

/// safe log for Eigen arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
safe_log(const T& x)
{
    return (abs(x) > NEAR_ZERO).select(log(abs(x)),MIN_FLT);
};

/// relu
template<typename T>
std::enable_if_t<std::is_scalar_v<T>, T> 
relu(const T& x)
{
    return max(x, T(0.01)); 
};

/// relu for Eigen arrays
template<typename T>
std::enable_if_t<std::is_base_of_v<Eigen::ArrayBase<T>, T>, T> 
relu(const T& x)
{
    // return (x > 0).select(x, T::Zero(x.size())+T(0.01));
    return (x > 0).select(x, T::Zero(x.size()));
};

// template<typename T, typename U, typename V> struct BinaryOperator;

template<typename T, typename U=T, typename V=T>
struct BinaryOperator
{
    std::string name;
    int complexity;
    const function<T(U,V)> f;
    const function<array<T,2>(U,V)> df;

    inline BinaryOperator(string n, 
                          int c, 
                          const function<T(U,V)>& f, 
                          const function<array<T,2>(U,V)>& df
                         ): name(n),  complexity(c), f(f), df(df)  {}
};

template<typename T, typename U=T>
struct UnaryOperator
{
    std::string name;
    int complexity;
    const function<T(U)> f;
    const function<array<T,1>(U)> df;

    inline UnaryOperator(string n, 
                          int c, 
                          const function<T(U)>& f, 
                          const function<array<T,1>(U)>& df
                         ): name(n),  complexity(c), f(f), df(df)  {}
};

/* Binary Operators */

template<typename T> 
struct Add : public BinaryOperator<T>
{
    Add(): BinaryOperator<T>("ADD", 2, std::plus<T>(), d_add<T>) {}
};

template<typename T> 
struct Sub : public BinaryOperator<T>
{
    Sub(): BinaryOperator<T>("SUB", 2, std::minus<T>(), d_sub<T>) {}
};

template<typename T> 
struct Times : public BinaryOperator<T>
{
    Times(): BinaryOperator<T>("TIMES", 3, std::multiplies<T>(), d_times<T>) {}
};

template<typename T> 
struct Div : public BinaryOperator<T>
{
    Div(): BinaryOperator<T>("DIV", 4, std::divides<T>(), d_div<T>) {}
};


template<typename T> 
struct Pow : public BinaryOperator<T>
{
    Pow(): BinaryOperator<T>("POW", 7, 
        [](const T& a, const T& b){return pow(a,b);}, 
        d_pow<T>
    ) {}
};

/* Unary Operators */

template<typename T> 
struct Exp : public UnaryOperator<T>
{
    Exp(): UnaryOperator<T>("EXP", 7, 
                            [](const T& x){ return exp(x); }, 
                            [](const T& x) -> array<T,1>{ return {exp(x)}; } 
    ) {}
};

template<typename T> 
struct SafeLog : public UnaryOperator<T> 
{
    SafeLog(): UnaryOperator<T>("LOG", 4, safe_log<T>, d_safe_log<T>) {}
};

template<typename T> 
struct Sin : public UnaryOperator<T>
{
    Sin(): UnaryOperator<T>("SIN", 9,  
                            [](const T& x){ return sin(x); }, 
                            [](const T& x) -> array<T,1>{ return {-cos(x)}; }
    ) {}
};

template<typename T> 
struct Cos : public UnaryOperator<T>
{
    Cos(): UnaryOperator<T>("COS", 9, 
                            [](const T& x){ return cos(x); }, 
                            [](const T& x) -> array<T,1>{ return {sin(x)}; }
    ) {}
};

template<typename T> 
struct Tanh : public UnaryOperator<T>
{
    Tanh(): UnaryOperator<T>("TANH", 9,  
                    [](const T& x){ return tanh(x); }, 
                    [](const T& x) -> array<T,1>{return {1-pow(tanh(x), 2)}; } 
    ) {} 
};

template<typename T> 
struct Sqrt : public UnaryOperator<T>
{
    Sqrt(): UnaryOperator<T>("SQRT", 5, 
                    [](const T& x){ return sqrt(x); }, 
                    [](const T& x) -> array<T,1>{return {x/ (2*sqrt(abs(x)))}; } 
    ){}
};

template<typename T> 
struct Square : public UnaryOperator<T>
{
    Square(): UnaryOperator<T>("SQUARE", 4, 
                               [](const T& x){ return pow(x, 2); },
                               [](const T& x) -> array<T,1>{return {2*x}; }
    ){}
};

template<typename T> 
struct Cube : public UnaryOperator<T>
{
    Cube(): UnaryOperator<T>("CUBE", 4, 
        [](const T& x){ return pow(x, 3); },
        [](const T& x) -> array<T,1>{return {3*pow(x, T(2))}; }
    ){}
};

template<typename T> 
struct Logit : public UnaryOperator<T>
{
    Logit(): UnaryOperator<T>("LOGIT", 3, 
        [](const T& x){ return 1/(1+exp(-x)); },
        d_logit<T>
    ){}
};

template<typename T>
struct Relu : public UnaryOperator<T>
{
    Relu(): UnaryOperator<T>("RELU", 3, relu<T>, 
        d_relu<T>
    ) {}  
};

template<typename T, typename U>
inline T lt( const U& A, const U& B) { return T(A < B); };

////////////////////////////////////////////////////////////////////////////////
/* (TODO): Reductions */

/* At a point x where there is exactly one function fi such that fi(x) is the median, the derivative of the median is indeed the derivative of fi (this is not the median of the derivatives). 
At a point where there are more than one functions equal to the median, the derivative of the median exists only if the derivatives of these functions are equal at that point. 
Otherwise, the derivative of the median does not exist at that point.
*/
template<typename T>
T d_median(const Array<T,-1,1>& v) 
{
    // instantiate a vector
    vector<float> x(v.size());
    x.assign(v.data(),v.data()+v.size());
    // middle element
    size_t n = x.size()/2;
    // sort nth element of array
    nth_element(x.begin(),x.begin()+n,x.end());
    // if evenly sized, return average of middle two elements
    if (x.size() % 2 == 0) {
        nth_element(x.begin(),x.begin()+n-1,x.end());
        return (x[n] + x[n-1]) / 2;
    }
    // otherwise return middle element
    else
        return x[n];
}


// /// calculate variance when mean provided
// float variance(const ArrayXf& v, float mean) 
// {
//     ArrayXf tmp = mean*ArrayXf::Ones(v.size());
//     return pow((v - tmp), 2).mean();
// }

// /// calculate variance
// float variance(const ArrayXf& v) 
// {
//     float mean = v.mean();
//     return variance(v, mean);
// }


////////////////////////////////////////////////////////////////////////////////
/* Operator creation routines */
// template<typename T, typename U=T, typename V=T>
// NodeBase* make_node(string op_name)
// {
//     auto op = make_op_map<T,U,V>()()[op_name];

//     if (op->name == "best_split")
//         return new SplitNode<T(T,T)>(op->name, op->f, op->df);
//     else if (op->name == "arg_split")
//         return new SplitNode<T(U,T,T)>(op->name, op->f, op->df);
//     else
//         return new WeightedDxNode<T(U,V)>(op->name, op->f, op->df);

// }
// todo: make these return maps?
// typedef map<string, std::function<shared_ptr<BinaryOperator<T>>()>> BinOpMaker;

/// define binary operators. Right now they are just defined for ArrayXf data.
template <typename T> struct make_binary_operators;
// vector<shared_ptr<BinaryOperator<T>>> make_binary_operators(vector<string>&);
template <typename T> struct make_unary_operators; 
// vector<shared_ptr<UnaryOperator<T>>> make_unary_operators(vector<string>&);

// template<typename T>
// vector<shared_ptr<BinaryOperator<T>>> make_binary_operators(
//         vector<string>& op_names)
// { 
//     return {}; 
// };

template<>
struct make_binary_operators<ArrayXf>
{
    typedef map<string, std::function<shared_ptr<BinaryOperator<ArrayXf>>()>> BinOpMap;

    vector<shared_ptr<BinaryOperator<ArrayXf>>> operator()(vector<string>& op_names)
    {
        vector<shared_ptr<BinaryOperator<ArrayXf>>> binary_operators;

        BinOpMap binary_op_map = {
                {"ADD",     make_shared<Add<ArrayXf>> },
                {"SUB",     make_shared<Sub<ArrayXf>> },
                {"TIMES",   make_shared<Times<ArrayXf>> },
                {"DIV",     make_shared<Div<ArrayXf>> },
                {"POW",     make_shared<Pow<ArrayXf>> },
        };
        // if op names is empty, return all nodes
        if (op_names.empty())
        {
            binary_operators.resize(binary_op_map.size());
            transform(binary_op_map.begin(), binary_op_map.end(), 
                    binary_operators.begin(),
                    [](const auto& bom){return bom.second();});
            return binary_operators;
        }
        for (const auto& op: op_names)
        {
            if (binary_op_map.find(op) != binary_op_map.end())
                binary_operators.push_back(binary_op_map[op]());
        }
        return binary_operators;
    };
};

/// define unary operators. Right now they are just defined for ArrayXf data.
// template<typename T>
// vector<shared_ptr<UnaryOperator<T>>> make_unary_operators(
//         vector<string>& op_names)
// { 
//     return {}; 
// };

template<>
struct make_unary_operators<ArrayXf>
{
    vector<shared_ptr<UnaryOperator<ArrayXf>>> operator()(vector<string>& op_names)
    {
        vector<shared_ptr<UnaryOperator<ArrayXf>>> all_ops = {
                make_shared<Sin<ArrayXf>>(),
                make_shared<Cos<ArrayXf>>(),
                make_shared<Exp<ArrayXf>>(),
                make_shared<SafeLog<ArrayXf>>(),
                make_shared<Sqrt<ArrayXf>>(),
                make_shared<Square<ArrayXf>>(),
                make_shared<Cube<ArrayXf>>(),
                make_shared<Tanh<ArrayXf>>(),
                make_shared<Logit<ArrayXf>>(),
                make_shared<Relu<ArrayXf>>()
        };
        if (op_names.empty())
            return all_ops;

        vector<shared_ptr<UnaryOperator<ArrayXf>>> filtered_ops;
        std::copy_if(all_ops.begin(), all_ops.end(),
                    std::back_inserter(filtered_ops),
                    [&](const auto& o) { return in(op_names, o->name); }
                    );    
        return filtered_ops;
    };

};



} // Brush

#endif
