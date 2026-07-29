import numpy as np
import pytest

from algo_stack.supervised.svm import SVC, SVR


def _two_blobs(seed=0, n=60, sep=2.5):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=[-sep, 0.0], scale=0.5, size=(n, 2))
    X1 = rng.normal(loc=[sep, 0.0], scale=0.5, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def _three_blobs(seed=0, n=40):
    rng = np.random.default_rng(seed)
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.45, size=(n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_binary_svc_separates_blobs(kernel):
    X, y = _two_blobs()
    clf = SVC(C=1.0, kernel=kernel, max_iter=400, tol=1e-3, random_state=0).fit(X, y)
    assert clf.score(X, y) >= 0.95
    assert clf.support_vectors_.ndim == 2
    assert clf.dual_coef_.shape[0] == clf.support_vectors_.shape[0]
    assert isinstance(clf.intercept_, float)
    assert list(clf.classes_) == [0, 1]


def test_svc_decision_sign_matches_predict():
    X, y = _two_blobs()
    clf = SVC(kernel="linear", C=10.0, max_iter=300, random_state=1).fit(X, y)
    scores = clf.decision_function(X)
    pred = clf.predict(X)
    np.testing.assert_array_equal(pred, clf.classes_[(scores >= 0).astype(int)])


def test_multiclass_ovr():
    X, y = _three_blobs()
    clf = SVC(C=1.0, kernel="rbf", max_iter=400, random_state=0).fit(X, y)
    assert clf.score(X, y) >= 0.9
    scores = clf.decision_function(X)
    assert scores.shape == (X.shape[0], 3)
    assert clf.intercept_.shape == (3,)
    assert len(clf.classes_) == 3


def test_string_labels():
    X, y = _two_blobs(n=40)
    y = np.array(["neg" if t == 0 else "pos" for t in y])
    clf = SVC(kernel="linear", C=5.0, max_iter=200, random_state=0).fit(X, y)
    pred = clf.predict(X[:5])
    assert all(p in ("neg", "pos") for p in pred)


def test_svr_recovers_sine():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.5, 2.5, size=(100, 1))
    y = np.sin(X.ravel())
    reg = SVR(C=50.0, kernel="rbf").fit(X, y)
    grid = np.linspace(-2, 2, 40).reshape(-1, 1)
    pred = reg.predict(grid)
    assert np.max(np.abs(pred - np.sin(grid.ravel()))) < 0.25
    assert reg.support_vectors_.shape == X.shape
    assert reg.dual_coef_.shape == (X.shape[0],)


def test_svr_linear_almost_ols():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 2))
    true_w = np.array([1.5, -2.0])
    y = X @ true_w + 0.05 * rng.normal(size=80)
    # Large C → small λ → close to interpolating kernel ridge / OLS in feature space
    reg = SVR(C=1e3, kernel="linear").fit(X, y)
    assert reg.score(X, y) > 0.98


def test_invalid_kernel_raises():
    X, y = _two_blobs(n=10)
    with pytest.raises(ValueError):
        SVC(kernel="poly").fit(X, y)
    with pytest.raises(ValueError):
        SVR(kernel="poly").fit(X, y)


def test_gamma_scale_attribute():
    X, y = _two_blobs(n=20)
    clf = SVC(kernel="rbf", gamma="scale", max_iter=50, random_state=0).fit(X, y)
    assert clf.gamma_ > 0
    expected = 1.0 / (X.shape[1] * X.var())
    np.testing.assert_allclose(clf.gamma_, expected)
