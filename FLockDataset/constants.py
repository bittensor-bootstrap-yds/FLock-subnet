from pathlib import Path
from dataclasses import dataclass
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent
SUBNET_OWNER_HOTKEY = "5Cex1UGEN6GZBcSBkWXtrerQ6Zb7h8eD7oSe9eDyZmj4doWu" # EXAMPLE
SUBNET_UID = 257 # Testnet
NUM_UIDS = 2 ** 8 - 1
EVAL_SIZE = 50

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
                id=str(data.get('id', 0)),  
                bench=data.get('bench', 0)
            )
        except Exception:
            return None

# eval dataset huggingface
eval_namespace = "silassilas/base"
eval_commit = "main"
