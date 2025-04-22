from pathlib import Path
from dataclasses import dataclass
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent
SUBNET_OWNER_HOTKEY = "5Cex1UGEN6GZBcSBkWXtrerQ6Zb7h8eD7oSe9eDyZmj4doWu" # EXAMPLE
SUBNET_UID = 257 # Testnet
NUM_UIDS = 2 ** 8 - 1

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
                id=str(data.get('id', 0)),  # Convert to string
                bench=data.get('bench', 0)
            )
        except Exception:
            return None

CONSTANT_ALPHA = 0.2  # enhance vtrust
timestamp_epsilon = 0.04  # enhance vtrust
temperature = 0.08

# validator weight moving average term. alpha = 1-lr.
lr = 0.2

# eval dataset huggingface
eval_namespace = "xiaofengzi/flcok_test"
eval_commit = "main"
