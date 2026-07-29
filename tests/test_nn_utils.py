import numpy as np

from algo_stack.utils import activations, optim


def test_activation_roundtrip():
    z = np.array([[-2.0, 0.0, 3.0]])
    for name in ("relu", "tanh", "sigmoid"):
        a = activations.forward(name, z)
        g = activations.grad(name, a)
        assert g.shape == z.shape


def test_softmax_sums_to_one():
    z = np.array([[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]])
    p = activations.softmax(z)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_sgd_momentum_and_adam_step():
    p = [np.array([1.0, 2.0])]
    g = [np.array([0.5, -0.5])]
    for name in ("sgd", "momentum", "adam"):
        opt = optim.make_optimizer(name, learning_rate=0.1)
        params = [arr.copy() for arr in p]
        opt.reset(params)
        opt.step(params, g)
        assert not np.allclose(params[0], p[0])
