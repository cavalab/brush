#include "node.h"

namespace Brush {

ostream& operator<<(ostream& os, const NodeType& nt)
{
    os << NodeTypeName.at(nt);
    return os;
}

ostream& operator<<(ostream& os, const Node& n)
{
    os << n.get_name();
    return os;
}

/// @brief get the name of the node. 
/// @return name 
auto Node::get_name() const noexcept -> std::string 
{

    if (Is<NodeType::Terminal>(node_type))
        return feature;
    else if (Is<NodeType::Constant>(node_type))
    {
        return fmt::format("{:.3f}", W.at(0));
    }
    else if (Is<NodeType::SplitBest>(node_type))
        return fmt::format("SplitBest[{}>{:.3f}]", feature, W.at(0));
    else if (Is<NodeType::SplitOn>(node_type))
        return fmt::format("SplitOn[{:.3f}]", W.at(0));
    else
        return name;
}

////////////////////////////////////////
// serialization
// serialization for Node
// using json = nlohmann::json;

void to_json(json& j, const Node& p) 
{
    j = json{
        {"name", p.name},
        {"center_op", p.center_op}, 
        {"prob_change", p.prob_change}, 
        {"fixed", p.fixed}, 
        {"node_type", p.node_type}, 
        {"sig_hash", p.sig_hash}, 
        {"sig_dual_hash", p.sig_dual_hash}, 
        {"ret_type", p.ret_type}, 
        {"arg_types", p.arg_types}, 
        {"is_weighted", p.is_weighted}, 
        {"W", p.W}, 
        {"feature", p.feature}, 
        {"complete_hash", p.complete_hash} 
    };
}

using NT = NodeType;
void init_node_with_default_signature(Node& node)
{
    // if (Is<
    //     NT::Add,
    //     NT::Mul,
    //     NT::Min,
    //     NT::Max
    //     >(nt)) 
    //     return Signature<ArrayXf(ArrayXf,ArrayXf)>{};
    NT n = node.node_type;
    if (Is<
        NT::Abs,
        NT::Acos,
        NT::Asin,
        NT::Atan,
        NT::Cos,
        NT::Cosh,
        NT::Sin,
        NT::Sinh,
        NT::Tan,
        NT::Tanh,
        NT::Ceil,
        NT::Floor,
        NT::Exp,
        NT::Log,
        NT::Logabs,
        NT::Log1p,
        NT::Sqrt,
        NT::Sqrtabs,
        NT::Square,
        NT::Logistic,
        NT::CustomUnaryOp
        >(n)) 
    {
        node.set_signature<Signature<ArrayXf(ArrayXf)>>();
    }
    else if (Is<
        NT::Add,
        NT::Sub,
        NT::Mul,
        NT::Div,
        NT::Pow,
        NT::SplitBest,
        NT::CustomSplit
        >(n))
     {
        node.set_signature<Signature<ArrayXf(ArrayXf,ArrayXf)>>();
    }  
    else if (Is<
        NT::Min,
        NT::Max,
        NT::Mean,
        NT::Median,
        NT::Sum,
        NT::Prod,
        NT::Softmax
        >(n))
    {
        auto msg = fmt::format("Can't infer arguments for {} from json."
            " Please provide them.\n",n);
        HANDLE_ERROR_THROW(msg);
    }
    else if (Is<
        NT::SplitOn
        >(n))
    {
        node.set_signature<Signature<ArrayXf(ArrayXf,ArrayXf,ArrayXf)>>();
    }
    else{
        node.set_signature<Signature<ArrayXf()>>();
    }

}

void from_json(const json &j, Node& p)
{

    if (j.contains("node_type"))
        j.at("node_type").get_to(p.node_type);
    else
        HANDLE_ERROR_THROW("Node json must contain node_type");

    if (j.contains("name"))
        j.at("name").get_to(p.name);
    else        
        p.name = NodeTypeName[p.node_type];

    if (j.contains("center_op"))
        j.at("center_op").get_to(p.center_op);
    if (j.contains("fixed"))
        j.at("fixed").get_to(p.fixed);
    if (j.contains("feature"))
        j.at("feature").get_to(p.feature);
    if (j.contains("is_weighted"))
        j.at("is_weighted").get_to(p.is_weighted);
    else
        p.is_weighted=false;

    // if node has a ret_type and arg_types, get them. if not we need to make 
    // a signature
    bool make_signature=false;

    if (j.contains("ret_type"))
        j.at("ret_type").get_to(p.ret_type);
    else
        make_signature=true;
    if (j.contains("arg_types"))
        j.at("arg_types").get_to(p.arg_types);
    else
        make_signature=true;
    if (j.contains("sig_hash"))
        j.at("sig_hash").get_to(p.sig_hash);
    else
        make_signature=true;
    if (j.contains("sig_dual_hash"))
        j.at("sig_dual_hash").get_to(p.sig_dual_hash);
    else
        make_signature=true;
    if (j.contains("complete_hash"))
        j.at("complete_hash").get_to(p.complete_hash);
    else
        make_signature=true;

    if (make_signature){
        // fmt::print("using default signature...\n");
        init_node_with_default_signature(p);
    }
    p.init();
    // fmt::print("checking W\n");
    if (j.contains("W"))
        j.at("W").get_to(p.W);


    json new_json = p;
    // fmt::print("new node json: {}\n", new_json.dump(2));
}


}