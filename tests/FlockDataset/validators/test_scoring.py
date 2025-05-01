import numpy as np
from FLockDataset.validator.validator_utils import compute_score
from FLockDataset import constants

def test_high_loss_evaluation():
    loss = 9999999999999999
    benchmark_loss = 0.1
    expected_score = 0.0 
    score = compute_score(loss, benchmark_loss)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"

def test_zero_loss_evaluation():
    loss = 0 
    benchmark_loss = 0.1
    expected_score = 1.0  
    score = compute_score(loss, benchmark_loss)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"

def test_none_loss_evaluation():
    loss = None
    benchmark_loss = 0.1
    expected_score = 0.0
    score = compute_score(loss, benchmark_loss)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"

def test_zero_benchmark_evaluation():
    loss = 0.1
    benchmark_loss = 0
    expected_score = 1.0 / constants.NUM_UIDS  
    score = compute_score(loss, benchmark_loss)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"

def test_negative_benchmark_evaluation():
    loss = 0.1
    benchmark_loss = -0.1
    expected_score = 1.0 / constants.NUM_UIDS  
    score = compute_score(loss, benchmark_loss)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"

def test_none_benchmark_evaluation():
    loss = 0.1
    benchmark_loss = None
    expected_score = 1.0 / constants.NUM_UIDS  
    score = compute_score(loss, benchmark_loss)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"

def test_exponent_capping_high():
    loss = 1e-10
    benchmark_loss = 0.1
    score = compute_score(loss, benchmark_loss)
    assert np.isclose(score, 1.0, rtol=1e-7), f"Expected score close to 1.0, but got: {score}"

def test_exponent_capping_low():
    loss = 1e10
    benchmark_loss = 0.1
    capped_exp = -100
    expected_score = constants.NUM_UIDS**capped_exp
    score = compute_score(loss, benchmark_loss)
    assert score == expected_score, f"Expected score: {expected_score}, but got: {score}"

def test_lower_loss_than_benchmark():
    loss = 0.05
    benchmark_loss = 0.1
    exp = -loss * constants.DECAY_RATE / benchmark_loss
    expected_score = constants.NUM_UIDS**exp
    score = compute_score(loss, benchmark_loss)
    assert np.isclose(
        score, expected_score, rtol=1e-9
    ), f"Expected score: {expected_score}, but got: {score}"
