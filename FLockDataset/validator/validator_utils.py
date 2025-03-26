import bittensor as bt
import numpy as np
import typing
from FLockDataset import constants
from scipy import optimize


def iswin(loss_i, loss_j, block_i, block_j):
    loss_i = (1 - constants.timestamp_epsilon) * loss_i if block_i < block_j else loss_i
    loss_j = (1 - constants.timestamp_epsilon) * loss_j if block_j < block_i else loss_j
    return loss_i < loss_j


def compute_wins(
        uids: typing.List[int],
        scores_per_uid: typing.Dict,
        block: typing.Dict,
):
    blacklist_uids = []
    for i, uid_i in enumerate(uids):
        if uid_i in blacklist_uids:
            continue
        if scores_per_uid[uid_i] == 0:
            blacklist_uids.append(uid_i)
            continue

    whitelist_uids = [uid for uid in uids if uid not in blacklist_uids]

    wins = {uid: 0 for uid in uids}
    win_rate_1 = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(whitelist_uids):
        total_matches = 0
        block_i = block[uid_i]
        for j, uid_j in enumerate(whitelist_uids):
            if i == j:
                continue
            block_j = block[uid_j]

            scores_i = scores_per_uid[uid_i]
            scores_j = scores_per_uid[uid_j]
            wins[uid_i] += 1 if iswin(scores_i, scores_j, block_i, block_j) else 0
            total_matches += 1
        # Calculate win rate for uid i
        win_rate_1[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    weights = [(win_rate_1, 1)]
    weights_sum = sum([w for _, w in weights])
    weights = [(winrate_dict, weight / weights_sum) for winrate_dict, weight in weights]
    win_rate = {uid: sum([win_rate_dict[uid] * weight for win_rate_dict, weight in weights]) for uid in uids}
    return wins, win_rate


def adjust_for_vtrust(weights: np.ndarray, consensus: np.ndarray, vtrust_min: float = 0.5):
    """
    Interpolate between the current weight and the normalized consensus weights so that the
    vtrust does not fall below vturst_min, assuming the consensus does not change.
    """
    vtrust_loss_desired = 1 - vtrust_min

    # If the predicted vtrust is already above vtrust_min, then just return the current weights.
    orig_vtrust_loss = np.maximum(0.0, weights - consensus).sum()
    if orig_vtrust_loss <= vtrust_loss_desired:
        bt.logging.info("Weights already satisfy vtrust_min. {} >= {}.".format(1 - orig_vtrust_loss, vtrust_min))
        return weights

    # If maximum vtrust allowable by the current consensus is less that vtrust_min, then choose the smallest lambda
    # that still maximizes the predicted vtrust. Otherwise, find lambda that achieves vtrust_min.
    vtrust_loss_min = 1 - np.sum(consensus)
    if vtrust_loss_min > vtrust_loss_desired:
        bt.logging.info(
            "Maximum possible vtrust with current consensus is less than vtrust_min. {} < {}.".format(
                1 - vtrust_loss_min, vtrust_min
            )
        )
        vtrust_loss_desired = 1.05 * vtrust_loss_min

    # We could solve this with a LP, but just do rootfinding with scipy.
    consensus_normalized = consensus / np.sum(consensus)

    def fn(lam: float):
        new_weights = (1 - lam) * weights + lam * consensus_normalized
        vtrust_loss = np.maximum(0.0, new_weights - consensus).sum()
        return vtrust_loss - vtrust_loss_desired

    sol = optimize.root_scalar(fn, bracket=[0, 1], method="brentq")
    lam_opt = sol.root

    new_weights = (1 - lam_opt) * weights + lam_opt * consensus_normalized
    vtrust_pred = np.minimum(weights, consensus).sum()
    bt.logging.info("Interpolated weights to satisfy vtrust_min. {} -> {}.".format(1 - orig_vtrust_loss, vtrust_pred))
    return new_weights


def compute_score(loss, benchmark_loss):
    exp = -loss / benchmark_loss
    return constants.NUM_UIDS ** exp

