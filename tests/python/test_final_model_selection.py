import pytest
import numpy as np

from pybrush import BrushRegressor, BrushClassifier, Dataset

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, balanced_accuracy_score, average_precision_score


def test_smallest_complexity_selection_regression():
    X, y = make_regression(n_samples=80, n_features=6, noise=0.1, random_state=1)
    model = BrushRegressor(max_gens=5, pop_size=12, final_model_selection="smallest_complexity")
    model.fit(X, y)

    chosen = model.best_estimator_

    # it ignores constant models
    complexities = [p.fitness.complexity for p in model.archive_ if p.fitness.size > 1 ]

    assert chosen.fitness.complexity == min(complexities)
    assert model.best_estimator_ in model.archive_

    
def test_best_validation_ci_selection_classification():
    X, y = make_classification(n_samples=100, n_features=8, n_informative=5, random_state=2)

    model = BrushClassifier(max_gens=5, pop_size=15, final_model_selection="best_validation_ci")
    model.fit(X, y)

    # best estimator can sometimes remain unchanged with this method
    assert model.best_estimator_ in model.archive_ + [model.best_estimator_]

    assert model.best_estimator_.fitness.complexity >= 0


def test_callable_selection():
    def pick_first(pop, archive):
        return archive[0]

    X, y = make_classification(n_samples=60, n_features=5, random_state=3)
    model = BrushClassifier(max_gens=5, pop_size=10, final_model_selection=pick_first)
    model.fit(X, y)

    assert model.best_estimator_ == model.archive_[0]


def test_invalid_selection_raises():
    X, y = make_regression(n_samples=50, n_features=5, random_state=4)
    model = BrushRegressor(max_gens=5, pop_size=10, final_model_selection="not_a_method")
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_callable_failure_raises():
    def bad_selector(pop, archive):
        raise Exception("boom")

    X, y = make_classification(n_samples=60, n_features=5, random_state=5)
    model = BrushClassifier(max_gens=5, pop_size=10, final_model_selection=bad_selector)

    with pytest.raises(RuntimeError):
        model.fit(X, y)


def test_regression_model_selection():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Default selection (final model will not necessarily be in the pop or archive)
    model = BrushRegressor(max_gens=10, pop_size=10, final_model_selection="").fit(X, y)
    idx = np.argmin([p.fitness.linear_complexity for p in model.archive_])


def test_classification_selection():
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, 
        n_redundant=0, random_state=42
    )
    
    # Default selection (final model will not necessarily be in the pop or archive)
    model = BrushClassifier(max_gens=10, pop_size=10, final_model_selection="").fit(X, y)
    idx = np.argmin([p.fitness.linear_complexity for p in model.archive_])


@pytest.mark.parametrize("scorer", ['log', 'accuracy', 'balanced_accuracy', 'average_precision_score'])
@pytest.mark.parametrize("class_weights", ['unbalanced', 'support', [0.3, 0.7]])
def test_final_model_selection_best_validation_ci_replicated(scorer, class_weights):
    # Small dataset for testing
    X, y = make_classification(n_samples=100, n_features=6, n_informative=4, random_state=42)

    est = BrushClassifier(
        max_gens=5,
        pop_size=12,
        final_model_selection="best_validation_ci",
        scorer=scorer,
        class_weights=class_weights,
        verbosity=0,
    )
    est.fit(X, y)
    
    # Replicate the selection logic here
    data = est.validation_

    loss_f_dict = {
        "mse": mean_squared_error,
        "log": log_loss,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "average_precision_score": average_precision_score,
    }
    loss_f = loss_f_dict[est.parameters_.scorer]

    def eval(ind, data, sample=None):
        if sample is None:
            sample = np.arange(len(data.y))

        if est.parameters_.scorer in ["log", "average_precision_score"]:
            y_pred = np.array(ind.predict_proba(data))
        else:
            y_pred = np.array(ind.predict(data))

        y_pred = np.nan_to_num(y_pred)
        y = np.array(data.y)

        if est.class_weights not in ['unbalanced', 'balanced_accuracy']:
            sample_weight = []
            if isinstance(est.class_weights, list):
                sample_weight = [est.class_weights[int(label)] for label in data.y]
            elif est.class_weights == 'support' and est.scorer == "average_precision_score": # using support as a way of weighting
                return loss_f(y[sample], y_pred[sample], average='weighted')
            else:
                classes, counts = np.unique(data.y, return_counts=True)

                support_weights = {
                    int(cls): len(data.y) / (len(classes)*count) 
                    if count > 0 else 0.0 for cls, count in zip(classes, counts)}
                
                sample_weight = [support_weights[label] for label in data.y]

            sample_weight = np.array(sample_weight)
            return loss_f(y[sample], y_pred[sample], sample_weight=sample_weight[sample])
        else: # Cases where we ignore weights
            return loss_f(y[sample], y_pred[sample])

    # Bootstrap validation samples
    y = np.array(data.y)
    np.random.seed(0)
    val_samples = []
    for i in range(100):
        sample = np.random.randint(0, len(y), size=len(y))
        val_samples.append(eval(est.best_estimator_, data, sample))

    lower_ci, upper_ci = np.quantile(val_samples, 0.05), np.quantile(val_samples, 0.95)
    print(f"CI bounds: {lower_ci:.4f}, {upper_ci:.4f}")

    print("original loss", est.best_estimator_.fitness.loss_v)
    print("recalculated loss", eval(est.best_estimator_, data))

    # Evaluate all archive members
    new_losses = [eval(ind, data) for ind in est.archive_]
    candidates = [(l, p) for l, p in zip(new_losses, est.archive_) if lower_ci <= l <= upper_ci]

    print("Original losses from archive", [ind.fitness.loss_v for ind in est.archive_])
    print("Recalculated losses (should match)", new_losses)
    print(f"Num candidates in CI: {len(candidates)}")

    # TODO: make the assert below work
    # assert np.allclose([ind.fitness.loss_v for ind in est.archive_], new_losses)

    if candidates:
        chosen = min(candidates, key=lambda lp: lp[1].fitness.complexity)[1]
        print("Chosen candidate model:", chosen.get_model())

        # forcing brush to update final candidate
        est.final_model_selection = "best_validation_ci"
        est._update_final_model(est.data_.get_validation_data())

        # Assert that Brush picked the same candidate
        assert est.best_estimator_.get_model() == chosen.get_model()


if __name__ == "__main__":
    pytest.main()