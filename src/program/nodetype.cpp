#include "nodetype.h"

namespace Brush{

std::map<std::string, NodeType> NodeNameType = {
    //arithmetic
    {"Add", NodeType::Add},
    {"Sub", NodeType::Sub},
    {"Mul", NodeType::Mul},
    {"Div", NodeType::Div},
    /* {"Aq", NodeType::Aq}, */
    {"Abs", NodeType::Abs},
    {"Acos", NodeType::Acos},
    {"Asin", NodeType::Asin},
    {"Atan", NodeType::Atan},
    {"Cos", NodeType::Cos},
    {"Cosh", NodeType::Cosh},
    {"Sin", NodeType::Sin},
    {"Sinh", NodeType::Sinh},
    {"Tan", NodeType::Tan},
    {"Tanh", NodeType::Tanh},
    {"Ceil", NodeType::Ceil},
    {"Floor", NodeType::Floor},
    {"Exp", NodeType::Exp},
    {"Log", NodeType::Log},
    {"Logabs", NodeType::Logabs},
    {"Log1p", NodeType::Log1p},
    {"Sqrt", NodeType::Sqrt},
    {"Sqrtabs", NodeType::Sqrtabs},
    {"Square", NodeType::Square},
    {"Pow", NodeType::Pow},
    {"Logistic", NodeType::Logistic},

    // logic; not sure these will make it in
    // {"And", NodeType::And},
    // {"Or", NodeType::Or},
    // {"Not", NodeType::Not},
    // {"Xor", NodeType::Xor},

    // decision (same)
    /* {"Equals", NodeType::Equals}, */
    /* {"LessThan", NodeType::LessThan}, */
    /* {"GreaterThan", NodeType::GreaterThan}, */
    /* {"Leq", NodeType::Leq}, */
    /* {"Geq", NodeType::Geq}, */

    // reductions
    {"Min", NodeType::Min},
    {"Max", NodeType::Max},
    {"Mean", NodeType::Mean},
    {"Median", NodeType::Median},
    {"Count", NodeType::Count},
    {"Sum", NodeType::Sum},
    {"Prod", NodeType::Prod},
    {"ArgMax", NodeType::ArgMax},

    // transforms
    {"Softmax", NodeType::Softmax},

    // timing masks
    {"Before", NodeType::Before},
    {"After", NodeType::After},
    {"During", NodeType::During},

    //split
    {"SplitBest", NodeType::SplitBest},
    {"SplitOn", NodeType::SplitOn},

    //mean label
    {"MeanLabel", NodeType::MeanLabel},

    // leaves
    {"Constant", NodeType::Constant},
    {"Terminal", NodeType::Terminal},

    // custom
    {"CustomUnaryOp", NodeType::CustomUnaryOp},
    {"CustomBinaryOp", NodeType::CustomBinaryOp},
    {"CustomSplit", NodeType::CustomSplit},
};

std::map<NodeType,std::string> NodeTypeName = Util::reverse_map(NodeNameType);
} // Brush
