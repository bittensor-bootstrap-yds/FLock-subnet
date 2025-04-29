import bittensor as bt
from FLockDataset import constants

def compute_score(loss, benchmark_loss):
    """
    Compute the score based on the loss and benchmark loss.
    
    Args:
        loss: The loss value to evaluate
        benchmark_loss: The benchmark loss to compare against
        
    Returns:
        float: Score value between 0 and 1
    """
    if loss is None:
        bt.logging.warning("Loss is None, returning score of 0")
        return 0
        
    if benchmark_loss is None or benchmark_loss <= 0:
        bt.logging.error(f"Invalid benchmark_loss ({benchmark_loss}). Returning baseline score.")
        return 1.0 / constants.NUM_UIDS
        
    exp = -loss * constants.DECAY_RATE / benchmark_loss
    
    if exp > 100:  
        bt.logging.warning(f"Exponent {exp} is too large, capping to 100")
        exp = 100
    elif exp < -100:  
        bt.logging.warning(f"Exponent {exp} is too small, capping to -100")
        exp = -100
        
    return constants.NUM_UIDS**exp
