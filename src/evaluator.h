/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef EVALUATOR_H 
#define EVALUATOR_H

//internal includes
#include "init.h"
#include "program/node.h"
#include "program/nodetype.h"
#include "program/tree_node.h"
// #include "program/program.h"
#include "util/utils.h"
#include "util/rnd.h"
#include "params.h"
#include <utility>
#include <optional>

using namespace Brush::Data;
using namespace Brush::Util; 
using Brush::Node;
using Brush::DataType;
using std::type_index; 

namespace Brush
{

// template <typename DTable>
// class OPERON_EXPORT MinimumDescriptionLengthEvaluator final : public Evaluator<DTable> {
//     using Base = Evaluator<DTable>;

// public:
//     explicit MinimumDescriptionLengthEvaluator(Operon::Problem& problem, DTable const& dtable)
//         : Base(problem, dtable, sse_)
//         , sigma_(1, 1) // assume unit variance by default
//     {
//     }

//     auto SetSigma(std::vector<Operon::Scalar> sigma) { sigma_ = std::move(sigma); }

//     auto
//     operator()(Operon::RandomGenerator& /*random*/, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;

// private:
//     Operon::SSE sse_;
//     mutable std::vector<Operon::Scalar> sigma_;
// };

} // namespace Brush
#endif
