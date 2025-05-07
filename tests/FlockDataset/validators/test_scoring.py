import numpy as np
from flockoff.validator.validator_utils import compute_score
from flockoff import constants


def test_pow_8():
    benchmark_loss = 0.16
    power = 8
    loss = 0.15
    score = compute_score(loss, benchmark_loss, power)

    assert np.isclose(
        score, 0.683006389, rtol=1e-6
    ), f"Expected score: 0.683006389, but got: {score}"


def test_high_loss_evaluation():
    loss = 9999999999999999
    benchmark_loss = 0.1
    power = 2  # Adding power parameter
    expected_score = 0.0
    score = compute_score(loss, benchmark_loss, power)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"


def test_zero_loss_evaluation():
    loss = 0
    benchmark_loss = 0.1
    power = 2  # Adding power parameter
    expected_score = 1.0
    score = compute_score(loss, benchmark_loss, power)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"


def test_none_loss_evaluation():
    loss = None
    benchmark_loss = 0.1
    power = 2  # Adding power parameter
    expected_score = 0.0
    score = compute_score(loss, benchmark_loss, power)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"


def test_zero_benchmark_evaluation():
    loss = 0.1
    benchmark_loss = 0
    power = 2  # Adding power parameter
    expected_score = constants.DEFAULT_SCORE
    score = compute_score(loss, benchmark_loss, power)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"


def test_negative_benchmark_evaluation():
    loss = 0.1
    benchmark_loss = -0.1
    power = 2  # Adding power parameter
    expected_score = constants.DEFAULT_SCORE
    score = compute_score(loss, benchmark_loss, power)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"


def test_none_benchmark_evaluation():
    loss = 0.1
    benchmark_loss = None
    power = 2  # Adding power parameter
    expected_score = constants.DEFAULT_SCORE
    score = compute_score(loss, benchmark_loss, power)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"


def test_invalid_power():
    loss = 0.1
    benchmark_loss = 0.1
    power = 3  # Odd number, should trigger error
    expected_score = constants.DEFAULT_SCORE
    score = compute_score(loss, benchmark_loss, power)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"


def test_none_power():
    loss = 0.1
    benchmark_loss = 0.1
    power = None  # None value, should trigger error
    expected_score = constants.DEFAULT_SCORE
    score = compute_score(loss, benchmark_loss, power)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"


def test_different_power_values():
    loss = 0.05
    benchmark_loss = 0.1
    power_values = [2, 4, 6, 8]

    for power in power_values:
        center_point = (power - 1) / (power + 1) * (1 / benchmark_loss) ** power
        expected_score = 1 / (1 + center_point * loss**power)
        score = compute_score(loss, benchmark_loss, power)
        assert np.isclose(
            score, expected_score, rtol=1e-9
        ), f"For power={power}, expected score: {expected_score}, but got: {score}"
