import bittensor as bt
from FLockDataset import constants


def compute_score(loss, benchmark_loss):
    if loss is None:
        bt.logging.warning("Loss is None, returning score of 0")
        return 0

    exp = -loss / benchmark_loss
    return constants.NUM_UIDS**exp
