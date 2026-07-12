import numpy as np

from algo_stack.unsupervised.autoencoder import Autoencoder


def _low_rank_data(n=300, n_features=15, rank=3):
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(n, rank))
    W = rng.normal(size=(rank, n_features))
    return latent @ W + 0.05 * rng.normal(size=(n, n_features))


def test_reconstruction_improves():
    X = _low_rank_data()
    ae = Autoencoder(
        encoding_dim=3, hidden_dim=12, learning_rate=0.01, n_epochs=150, random_state=0
    ).fit(X)
    recon = ae.reconstruct(X)
    mse = float(np.mean((X - recon) ** 2))
    assert mse < 0.5
    assert ae.loss_curve_[-1] < ae.loss_curve_[0]


def test_transform_shape():
    X = _low_rank_data(n=50, n_features=10, rank=2)
    ae = Autoencoder(encoding_dim=2, n_epochs=50, random_state=0).fit(X)
    Z = ae.transform(X)
    assert Z.shape == (50, 2)


def test_fit_transform():
    X = _low_rank_data(n=40, n_features=8, rank=2)
    ae = Autoencoder(encoding_dim=2, n_epochs=30, random_state=0)
    Z = ae.fit_transform(X)
    assert Z.shape == (40, 2)
