/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#include "evaluator.h"
namespace Brush
{
    
// template<> auto OPERON_EXPORT
// MinimumDescriptionLengthEvaluator<DefaultDispatch>::operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType {
//     auto const& problem = Evaluator::GetProblem();
//     auto const range = problem.TrainingRange();
//     auto const& dataset = problem.GetDataset();
//     auto const& nodes = ind.Genotype.Nodes();
//     auto const& dtable = Evaluator::GetDispatchTable();

//     auto const* optimizer = Evaluator::GetOptimizer();
//     EXPECT(optimizer != nullptr);

//     // this call will optimize the tree coefficients and compute the SSE
//     auto const& tree = ind.Genotype;
//     Operon::Interpreter<Operon::Scalar, DefaultDispatch> interpreter{dtable, dataset, ind.Genotype};
//     auto summary = optimizer->Optimize(rng, tree);
//     auto parameters = summary.Success ? summary.FinalParameters : tree.GetCoefficients();
//     auto const p { static_cast<double>(parameters.size()) };

//     std::vector<Operon::Scalar> buffer;
//     if (buf.size() < range.Size()) {
//         buffer.resize(range.Size());
//         buf = Operon::Span<Operon::Scalar>(buffer);
//     }
//     interpreter.Evaluate(parameters, range, buf);

//     auto estimatedValues = buf;
//     auto targetValues    = problem.TargetValues(range);

//     // codelength of the complexity
//     // count number of unique functions
//     // - count weight * variable as three nodes
//     // - compute complexity c of the remaining numerical values
//     //   (that are not part of the coefficients that are optimized)
//     Operon::Set<Operon::Hash> uniqueFunctions; // to count the number of unique functions
//     auto k{0.0}; // number of nodes
//     auto cComplexity { 0.0 };

//     // codelength of the parameters
//     Eigen::Matrix<Operon::Scalar, -1, -1> j = interpreter.JacRev(parameters, range); // jacobian
//     auto fm = optimizer->ComputeFisherMatrix(estimatedValues, {j.data(), static_cast<std::size_t>(j.size())}, sigma_);
//     auto ii = fm.diagonal().array();
//     ENSURE(ii.size() == p);

//     auto cParameters { 0.0 };
//     auto constexpr eps = std::numeric_limits<Operon::Scalar>::epsilon(); // machine epsilon for zero comparison

//     for (auto i = 0, j = 0; i < std::ssize(nodes); ++i) {
//         auto const& n = nodes[i];

//         // count the number of nodes and the number of unique operators
//         k += n.IsVariable() ? 3 : 1;
//         uniqueFunctions.insert(n.HashValue);

//         if (n.Optimize) {
//             // this branch computes the description length of the parameters to be optimized
//             auto const di = std::sqrt(12 / ii(j));
//             auto const ci = std::abs(parameters[j]);

//             if (!(std::isfinite(ci) && std::isfinite(di)) || ci / di < 1) {
//                 //ind.Genotype[i].Optimize = false;
//                 //auto const v = ind.Genotype[i].Value;
//                 //ind.Genotype[i].Value = 0;
//                 //auto fit = (*this)(rng, ind, buf);
//                 //ind.Genotype[i].Optimize = true;
//                 //ind.Genotype[i].Value = v;
//                 //return fit;
//             } else {
//                 cParameters += 0.5 * std::log(ii(j)) + std::log(ci);
//             }
//             ++j;
//         } else {
//             // this branch computes the description length of the remaining tree structure
//             if (std::abs(n.Value) < eps) { continue; }
//             cComplexity += std::log(std::abs(n.Value));
//         }
//     }

//     auto q { static_cast<double>(uniqueFunctions.size()) };
//     if (q > 0) { cComplexity += static_cast<double>(k) * std::log(q); }

//     cParameters -= p/2 * std::log(3);

//     auto cLikelihood = optimizer->ComputeLikelihood(estimatedValues, targetValues, sigma_);
//     auto mdl = cComplexity + cParameters + cLikelihood;
//     if (!std::isfinite(mdl)) { mdl = EvaluatorBase::ErrMax; }
//     return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(mdl) };
// }

} // namespace Brush
