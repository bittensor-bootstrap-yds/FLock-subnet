import numpy as np
from FLockDataset.validator.evaluation import compute_score


def test_evaluation():
    loss = 9999999999999999
    benchmark_loss = 0.1
    expected_score = 0.0

    score = compute_score(loss, benchmark_loss)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"
