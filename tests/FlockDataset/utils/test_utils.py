import pytest
import bittensor as bt
from FLockDataset.utils.chain import read_chain_commitment

key = "5Cex1UGEN6GZBcSBkWXtrerQ6Zb7h8eD7oSe9eDyZmj4doWu"

@pytest.fixture
def node():
    """Create a mock subtensor with local node connection"""
    return bt.subtensor("test")


def test_read_chain_commitment(node):
    """Test reading commitment data from another neuron on the chain"""
    subnet_uid = 257

    print(f"Reading data for key: {key}")

    comp = read_chain_commitment(key, node, subnet_uid)

    print(f"Read data type: {type(comp)}, value: {comp}")

    assert comp is not None, "Should return a valid commitment"
    assert comp.id == "42", "ID should be 42"
    assert comp.bench == 100, "Bench should be 100"
