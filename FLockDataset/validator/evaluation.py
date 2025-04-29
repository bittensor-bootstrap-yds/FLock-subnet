from FLockDataset.constants import NUM_UIDS


def compute_score(loss, benchmark_loss):
    exp = -loss / benchmark_loss
    return NUM_UIDS**exp
