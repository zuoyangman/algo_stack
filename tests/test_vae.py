import numpy as np

from algo_stack.unsupervised.vae import VariationalAutoencoder


def _low_rank_data(n=300, n_features=15, rank=3):
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(n, rank))
    W = rng.normal(size=(rank, n_features))
    return latent @ W + 0.05 * rng.normal(size=(n, n_features))


def test_reconstruction_improves():
    X = _low_rank_data()
    vae = VariationalAutoencoder(
        encoding_dim=3,
        hidden_dim=12,
        learning_rate=0.01,
        n_epochs=120,
        beta=0.1,
        random_state=0,
    ).fit(X)
    recon = vae.reconstruct(X)
    mse = float(np.mean((X - recon) ** 2))
    assert mse < 1.0
    assert vae.loss_curve_[-1] < vae.loss_curve_[0]


def test_transform_shape():
    X = _low_rank_data(n=50, n_features=10, rank=2)
    vae = VariationalAutoencoder(encoding_dim=2, n_epochs=40, random_state=0).fit(X)
    Z = vae.transform(X)
    assert Z.shape == (50, 2)


def test_sample_shape():
    X = _low_rank_data(n=40, n_features=8, rank=2)
    vae = VariationalAutoencoder(encoding_dim=2, n_epochs=30, random_state=0).fit(X)
    samples = vae.sample(7)
    assert samples.shape == (7, 8)


def test_fit_transform():
    X = _low_rank_data(n=40, n_features=8, rank=2)
    vae = VariationalAutoencoder(encoding_dim=2, n_epochs=30, random_state=0)
    Z = vae.fit_transform(X)
    assert Z.shape == (40, 2)
