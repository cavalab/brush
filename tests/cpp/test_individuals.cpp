// TODO: test predict, predict proba, fit.

// TODO: test parent_id and id

#include "testsHeader.h"

using namespace Brush;
using namespace Brush::Pop;

TEST(Individual, FitAndPredictRegression)
{
	MatrixXf X(4, 2);
	ArrayXf y(4);

	X << 1.0, 2.0,
		 2.0, 1.0,
		 3.0, 0.5,
		 4.0, 1.5;
	y << 3.0, 3.0, 3.5, 5.5;

	Dataset data(X, y);
	SearchSpace ss(data);

    // We must have a SearchSpace reference, so the operator ret-type checks dont 
    // fail even when feature names look right --- node metadata is consistent.
	Parameters params;
	RegressorProgram prg = ss.make_regressor(0, 0, params);
	Individual<PT::Regressor> ind(prg);

	ASSERT_FALSE(ind.get_is_fitted());
	ind.fit(data);
	ASSERT_TRUE(ind.get_is_fitted());

	auto y_pred = ind.predict(data);
	ASSERT_EQ(y_pred.size(), y.size());
}

TEST(Individual, PredictProbaBinaryClassifier)
{
	MatrixXf X(6, 2);
	ArrayXf y(6);

	X << 0.0, 1.0,
		 1.0, 0.0,
		 0.5, 0.5,
		 0.2, 0.8,
		 0.8, 0.2,
		 1.0, 1.0;
	y << 0.0, 1.0, 0.0, 1.0, 1.0, 0.0;

	Dataset data(X, y, {}, {}, {}, true);
	SearchSpace ss(data);

	Parameters params;
	params.set_n_classes(data.y);
	params.set_sample_weights(data.y);

	ClassifierProgram prg = ss.make_classifier(0, 0, params);
	Individual<PT::BinaryClassifier> ind(prg);

	ind.fit(data);
	auto prob = ind.predict_proba(data);
	ASSERT_EQ(prob.size(), y.size());
}

TEST(Individual, ParentIdAndId)
{
	Individual<PT::Regressor> p1;
	Individual<PT::Regressor> p2;
	Individual<PT::Regressor> child;

	p1.set_id(3);
	p2.set_id(7);
	child.set_id(11);

	child.set_parents(std::vector<Individual<PT::Regressor>>{p1, p2});

	ASSERT_EQ(child.id, 11u);
	ASSERT_EQ(child.parent_id.size(), 2u);
	ASSERT_EQ(child.parent_id.at(0), 3u);
	ASSERT_EQ(child.parent_id.at(1), 7u);
}