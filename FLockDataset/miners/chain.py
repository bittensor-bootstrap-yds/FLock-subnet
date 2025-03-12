import functools
import multiprocessing
import bittensor as bt
from typing import Optional, Any


def run_in_subprocess(func: functools.partial, ttl: int) -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """

    def wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
        try:
            result = func()
            queue.put(result)
        except (Exception, BaseException) as e:
            # Catch exceptions here to add them to the queue.
            queue.put(e)

    # Use "fork" (the default on all POSIX except macOS), because pickling doesn't seem
    # to work on "spawn".
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result


async def store_model_metadata(subtensor: bt.subtensor,
                               wallet: Optional[bt.wallet],
                               subnet_uid: str, data: str):
    """Stores model metadata on this subnet for a specific wallet."""
    if wallet is None:
        raise ValueError("No wallet available to write to the chain.")

    # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
    partial = functools.partial(
        subtensor.commit,
        wallet,
        subnet_uid,
        data,
    )
    run_in_subprocess(partial, 60)
