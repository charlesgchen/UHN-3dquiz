import numpy as np


def test_equal_geometric_probability_fusion_matches_normalized_square_root_product():
    components = np.asarray([
        [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
        [[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
    ], dtype=np.float64)
    weights = np.asarray([0.5, 0.5])

    fused_logits = np.tensordot(
        weights, np.log(components.clip(1e-12)), axes=(0, 0)
    )
    fused_logits -= fused_logits.max(axis=1, keepdims=True)
    actual = np.exp(fused_logits)
    actual /= actual.sum(axis=1, keepdims=True)

    expected = np.sqrt(components[0] * components[1])
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(actual, expected)
