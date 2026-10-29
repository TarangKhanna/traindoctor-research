"""The independent update rule must agree with calculus, not verifier outputs."""
import numpy as np

from experiments.state_recovery_v1.oracle import loss_and_grad


def test_manual_mlp_gradients_against_central_finite_differences():
    rng = np.random.default_rng(17)
    params = [rng.normal(size=(3, 8)), np.ones(3) * 3,
              rng.normal(size=(4, 3)), rng.normal(size=4)]
    x, y = rng.normal(size=(2, 8)), rng.normal(size=(2, 4))
    _, grads = loss_and_grad(params, x, y)
    eps = 1e-5
    for p, grad in zip(params, grads):
        for index in np.ndindex(p.shape):
            orig = p[index]
            p[index] = orig + eps
            hi, _ = loss_and_grad(params, x, y)
            p[index] = orig - eps
            lo, _ = loss_and_grad(params, x, y)
            p[index] = orig
            assert np.isclose(grad[index], (hi - lo) / (2 * eps), atol=1e-7, rtol=1e-7)


def test_manual_gradients_zero_for_inactive_relu():
    params = [np.zeros((3, 8)), -np.ones(3), np.ones((4, 3)), np.zeros(4)]
    _, grad = loss_and_grad(params, np.ones((2, 8)), np.ones((2, 4)))
    assert all(np.array_equal(v, np.zeros_like(v)) for v in grad[:3])
    assert np.array_equal(grad[3], -np.ones(4) / 2)
