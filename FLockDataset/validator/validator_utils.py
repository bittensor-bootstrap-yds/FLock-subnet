import bittensor as bt
from FLockDataset import constants

def compute_score(loss, benchmark_loss):
    if loss is None:
        bt.logging.warning("Loss is None, returning score of 0")
        return 0

    if benchmark_loss is None or benchmark_loss <= 0:
        bt.logging.error("Invalid benchmark_loss (%s). Returning score of 0.", benchmark_loss)
        return 0

    exp = -loss * constants.DECAY_RATE / benchmark_loss
    return constants.NUM_UIDS**exp
