import numpy as np
import pytest

from algo_stack.supervised.multinomial_nb import MultinomialNB


def _count_data(seed=0, n=150, d=20):
    rng = np.random.default_rng(seed)
    lam0 = np.linspace(2.5, 0.3, d)
    lam1 = np.linspace(0.3, 2.5, d)
    X0 = rng.poisson(lam=lam0, size=(n, d)).astype(float)
    X1 = rng.poisson(lam=lam1, size=(n, d)).astype(float)
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_binary_high_accuracy():
    X, y = _count_data()
    clf = MultinomialNB(alpha=1.0).fit(X, y)
    assert clf.feature_log_prob_.shape == (2, X.shape[1])
    assert clf.class_log_prior_.shape == (2,)
    assert clf.score(X, y) > 0.9
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_feature_log_prob_normalised():
    X, y = _count_data()
    clf = MultinomialNB(alpha=1.0).fit(X, y)
    probs = np.exp(clf.feature_log_prob_)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)


def test_alpha_smoothing_avoids_neg_inf():
    X = np.array([[2.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    clf = MultinomialNB(alpha=1.0).fit(X, y)
    assert np.all(np.isfinite(clf.feature_log_prob_))
    # Class 0 never saw feature 1, but smoothing → finite log-prob.
    assert clf.feature_log_prob_[0, 1] > -np.inf


def test_negative_features_raise():
    X = np.array([[1.0, -1.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        MultinomialNB().fit(X, y)


def test_uniform_prior():
    X, y = _count_data()
    clf = MultinomialNB(fit_prior=False).fit(X, y)
    np.testing.assert_allclose(clf.class_log_prior_, [-np.log(2)] * 2)


def test_predict_log_proba_matches_proba():
    X, y = _count_data(seed=1, n=40)
    clf = MultinomialNB().fit(X, y)
    np.testing.assert_allclose(
        np.exp(clf.predict_log_proba(X)), clf.predict_proba(X), atol=1e-10
    )
