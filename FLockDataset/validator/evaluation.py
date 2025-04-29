from FLockDataset.constants import NUM_UIDS


def compute_score(loss, benchmark_loss):
    if loss is None:
        return 0

    if benchmark_loss == 0:
        return 0

    exp = -loss / benchmark_loss
    return NUM_UIDS**exp
