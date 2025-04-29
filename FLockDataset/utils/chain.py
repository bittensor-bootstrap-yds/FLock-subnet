import json
import bittensor as bt
from FLockDataset.constants import Competition
from typing import Optional


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph) -> int:
    """Asserts the wallet is a registered miner and returns the miner's UID.

    Raises:
        ValueError: If the wallet is not registered.
    """
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"You are not registered. \nUse: \n`btcli s register --netuid {metagraph.netuid}` to register via burn \n or btcli s pow_register --netuid {metagraph.netuid} to register with a proof of work"
        )
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.success(
        f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}"
    )
    return uid


def write_chain_commitment(
    wallet: bt.wallet, node, subnet_uid: int, data: dict
) -> bool:
    """
    Writes JSON data to the chain commitment.

    Args:
        wallet: The wallet for signing the transaction
        node: The subtensor node to connect to
        subnet_uid: The subnet identifier
        data: Dictionary containing the JSON data to commit

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert dict to JSON string
        json_str = json.dumps(data)

        # Pass the string directly - let bittensor handle the encoding
        result = node.commit(wallet, subnet_uid, json_str)
        return result
    except Exception as e:
        print(f"Failed to write chain commitment: {str(e)}")
        return False


def read_chain_commitment(ss58, node, subnet_uid: int) -> Optional[Competition]:
    """
    Reads JSON data from the chain commitment and returns it as a typed object.

    Args:
        ss58: The SS58 address of the hotkey
        node: The subtensor node to connect to
        subnet_uid: The subnet identifier

    Returns:
        Competition: The parsed data as a typed object or None if not found
    """
    try:
        # Get metadata from chain
        metadata = bt.core.extrinsics.serving.get_metadata(node, subnet_uid, ss58)

        if not metadata:
            print(f"No metadata found for hotkey {ss58} on subnet {subnet_uid}")
            return None

        # Extract commitment data - this is a complex nested structure
        fields = metadata["info"]["fields"]

        if not fields or len(fields) == 0:
            print("No fields found in metadata")
            return None

        # The first field contains our commitment
        field = fields[0]

        if isinstance(field, tuple) and len(field) > 0:
            # Extract Raw24 data which contains our JSON
            if isinstance(field[0], dict) and "Raw24" in field[0]:
                # The Raw24 field contains a tuple of integer values (ASCII codes)
                raw_data = field[0]["Raw24"]
                if isinstance(raw_data, tuple) and len(raw_data) > 0:
                    # Convert ASCII codes to bytes then to string
                    byte_data = bytes(raw_data[0])
                    json_str = byte_data.decode("utf-8")

                    try:
                        data_dict = json.loads(json_str)
                        return Competition.from_dict(data_dict)
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON from chain string: {json_str}")
                        return None

        print(f"Could not extract data from the commitment structure")
        return None

    except Exception as e:
        print(f"Failed to read chain commitment: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
