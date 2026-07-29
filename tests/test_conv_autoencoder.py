import numpy as np

from algo_stack.unsupervised.conv_autoencoder import ConvAutoencoder


def _bar_images(n_per=20, size=8, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    for i in range(2 * n_per):
        img = rng.normal(0, 0.05, size=(size, size))
        if i < n_per:
            img[size // 2, :] += 1.0
        else:
            img[:, size // 2] += 1.0
        X.append(img)
    return np.asarray(X, dtype=np.float64)[:, None, :, :]


def test_reconstruction_improves():
    X = _bar_images()
    cae = ConvAutoencoder(
        encoding_dim=4,
        num_filters=4,
        learning_rate=0.1,
        n_epochs=50,
        batch_size=10,
        random_state=0,
    ).fit(X)
    recon = cae.reconstruct(X)
    mse = float(np.mean((X - recon) ** 2))
    assert mse < 0.15
    assert cae.loss_curve_[-1] < cae.loss_curve_[0]


def test_transform_shape():
    X = _bar_images(n_per=12)
    cae = ConvAutoencoder(encoding_dim=3, n_epochs=20, random_state=0).fit(X)
    Z = cae.transform(X)
    assert Z.shape == (24, 3)


def test_fit_transform_and_reconstruct_shape():
    X = _bar_images(n_per=10)
    cae = ConvAutoencoder(encoding_dim=4, n_epochs=15, random_state=1)
    Z = cae.fit_transform(X)
    assert Z.shape == (20, 4)
    recon = cae.reconstruct(X)
    assert recon.shape == X.shape


def test_weight_shapes():
    X = _bar_images(n_per=8)
    cae = ConvAutoencoder(num_filters=5, encoding_dim=2, n_epochs=5, random_state=0).fit(
        X
    )
    assert cae.enc_conv_W_.shape == (5, 1, 3, 3)
    assert cae.dec_conv2_W_.shape == (1, 5, 3, 3)
    assert cae.image_shape_ == (1, 8, 8)
