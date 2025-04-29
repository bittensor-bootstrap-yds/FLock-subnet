from pathlib import Path
from dataclasses import dataclass
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent
SUBNET_OWNER = "5FZGwrY9Ycz8m6fq5rpZCgoSrQWddb7SnZCr3rFU61auctG2"
SUBNET_UID = 257
NUM_UIDS = 2**8 - 1
EVAL_SIZE = 50
DECAY_RATE = 5


@dataclass
class Competition:
    """Class defining model parameters"""

    id: str
    repo: str
    bench: float

    @classmethod
    def from_dict(cls, data: dict) -> Optional["Competition"]:
        """Create a ChainCommitment from a dictionary"""
        if not data:
            return None

        try:
            return cls(id=str(data.get("id", 0)), repo=data.get("repo", 0), bench=data.get("bench", 0))
        except Exception:
            return None


# eval dataset huggingface
eval_commit = "main"
