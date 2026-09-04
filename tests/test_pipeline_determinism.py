"""Confirms the training pipeline is deterministic given a fixed seed - a
non-deterministic pipeline would make the reported CV/test metrics
unreproducible and any future debugging much harder."""
import numpy as np

from feature_engineering import load_and_split
from train import build_pipeline


def test_same_config_produces_identical_predictions_across_fits():
    X_train, X_test, y_train, y_test = load_and_split()

    pipe1 = build_pipeline("logreg", imbalance="class_weight", params={"C": 0.3})
    pipe1.fit(X_train, y_train)
    proba1 = pipe1.predict_proba(X_test)[:, 1]

    pipe2 = build_pipeline("logreg", imbalance="class_weight", params={"C": 0.3})
    pipe2.fit(X_train, y_train)
    proba2 = pipe2.predict_proba(X_test)[:, 1]

    np.testing.assert_array_equal(proba1, proba2)


def test_random_forest_is_deterministic_given_fixed_seed():
    """RandomForestClassifier uses n_jobs=-1 (parallel tree fitting/averaging),
    so bit-exact equality isn't guaranteed - float summation order can differ
    by run at the ~1e-16 (machine epsilon) level even with a fixed
    random_state. That's floating-point non-associativity, not real
    non-determinism, so this asserts near-equality (rtol=1e-8) rather than
    exact equality - tight enough to still catch a genuine seeding bug."""
    X_train, X_test, y_train, y_test = load_and_split()

    pipe1 = build_pipeline("rf", imbalance="class_weight", params={"n_estimators": 50})
    pipe1.fit(X_train, y_train)
    proba1 = pipe1.predict_proba(X_test)[:, 1]

    pipe2 = build_pipeline("rf", imbalance="class_weight", params={"n_estimators": 50})
    pipe2.fit(X_train, y_train)
    proba2 = pipe2.predict_proba(X_test)[:, 1]

    np.testing.assert_allclose(proba1, proba2, rtol=1e-8, atol=1e-12)
