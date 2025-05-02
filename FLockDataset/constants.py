from pathlib import Path
import bittensor as bt
from dataclasses import dataclass
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent
NUM_UIDS = 2**8 - 1
DECAY_RATE = 1
MIN_WEIGHT_THRESHOLD = 1e-4

def get_subnet_owner(is_testnet: bool = False) -> str:
    """
    Returns the subnet owner based on whether it's a testnet or mainnet.
    """
    if is_testnet:
        return "5FZGwrY9Ycz8m6fq5rpZCgoSrQWddb7SnZCr3rFU61auctG2"
    else:
        return "5DFcEniKrQRbCakLFGY3UqPL3ZbNnTQHp8LTvLfipWhE2Yfr"


@dataclass
class Competition:
    """Class defining model parameters"""

    id: str
    repo: str
    bench: float
    rows: int
    pow: int

    @classmethod
    def from_dict(cls, data: dict) -> Optional["Competition"]:
        """Create a ChainCommitment from a dictionary"""
        if not data:
            return None

        try:
            id_val = str(data.get("id", ""))
            repo_val = str(data.get("repo", ""))
            bench_val = float(data.get("bench", 0.0))
            rows_val = int(data.get("rows", 250))
            pow_val = int(data.get("pow", 0))
            return cls(id=id_val, repo=repo_val, bench=bench_val, rows=rows_val, pow=pow_val)
        except (TypeError, ValueError) as e:
            bt.logging.warning(f"Failed to parse Competition from dict: {e}")
            return None


# eval dataset huggingface
eval_commit = "main"
