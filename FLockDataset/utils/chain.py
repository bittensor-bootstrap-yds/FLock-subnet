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


def read_chain_commitment(ss58: str, node: bt.subtensor, subnet_uid: int) -> Optional[Competition]:
    """
    Reads JSON data from the chain commitment (RawN) fields, recombines all
    byte‐tuples, decodes, and returns a Competition, or None.
    """
    try:
        metadata = bt.core.extrinsics.serving.get_metadata(node, subnet_uid, ss58)
        if not metadata:
            print(f"No metadata found for hotkey {ss58} on subnet {subnet_uid}")
            return None

        fields = metadata.get("info", {}).get("fields", ())
        if not fields:
            print("No fields found in metadata")
            return None

        # look only at the first field‐tuple
        field = fields[0]
        if not (isinstance(field, tuple) and field and isinstance(field[0], dict)):
            print("Unrecognized field structure:", field)
            return None

        raw_dict = field[0]
        # find the RawN key
        raw_key = next((k for k in raw_dict if k.startswith("Raw")), None)
        if raw_key is None:
            print("No RawN entry in first field:", raw_dict.keys())
            return None

        raw_segments = raw_dict[raw_key]  # e.g. ((byte1,byte2…), (byte25,byte26…), …)
        print(f"🔍 Found {raw_key} with {len(raw_segments)} segment(s)")

        # flatten all segments into one bytes object
        parts: list[bytes] = []
        for idx, seg in enumerate(raw_segments):
            # seg may itself be a tuple of ints or a nested tuple-of-tuples
            if isinstance(seg, (bytes, bytearray)):
                parts.append(bytes(seg))
            elif isinstance(seg, (list, tuple)):
                # if seg[0] is also list/tuple, dive in one level
                if seg and isinstance(seg[0], (list, tuple)):
                    for inner in seg:
                        parts.append(bytes(inner))
                else:
                    parts.append(bytes(seg))
            else:
                print(f"  ⚠️ unexpected segment #{idx} type={type(seg)}:", seg)

        full_bytes = b"".join(parts)
        print(f"🔍 combined byte‐length:", len(full_bytes))

        try:
            json_str = full_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            print("Failed to UTF-8 decode:", e, full_bytes)
            return None

        print("🔍 decoded JSON:", json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            return None

        comp = Competition.from_dict(data)
        print("🔍 Competition.from_dict →", comp)
        return comp

    except Exception as e:
        print("Unhandled exception in read_chain_commitment:", e)
        import traceback; traceback.print_exc()
        return None
