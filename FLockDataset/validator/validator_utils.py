from FLockDataset import constants

def compute_score(loss, benchmark_loss):
    exp = -loss / benchmark_loss
    return constants.NUM_UIDS ** exp
