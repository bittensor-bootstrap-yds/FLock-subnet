from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# The uid for this subnet.
# testnet
SUBNET_UID = 257
# The start block of this subnet
ROOT_DIR = Path(__file__).parent.parent


@dataclass
class Competition:
    """Class defining model parameters"""
    id: str
    bench: float

    @classmethod
    def from_dict(cls, data: dict) -> Optional['Competition']:
        """Create a ChainCommitment from a dictionary"""
        if not data:
            return None
        
        try:
            return cls(
                id=data.get('id', 0),
                bench=data.get('bench', 0)
            )
        except Exception as e:
            print(f"Error creating ChainCommitment from dict: {e}")
            return None


CONSTANT_ALPHA = 0.2  # enhance vtrust
timestamp_epsilon = 0.04  # enhance vtrust
temperature = 0.08

# validator weight moving average term. alpha = 1-lr.
lr = 0.2

ORIGINAL_COMPETITION_ID = "f127"

# eval dataset huggingface
eval_namespace = "xiaofengzi/flcok_test"
eval_commit = "main"
