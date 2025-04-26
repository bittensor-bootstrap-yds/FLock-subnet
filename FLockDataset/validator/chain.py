import torch
import bittensor as bt
from FLockDataset.miners.data import ModelId, ModelMetadata
from typing import Optional
from typing import Optional, Tuple, Union
from bittensor.core.extrinsics.set_weights import set_weights_extrinsic


def retrieve_model_metadata(subtensor: bt.subtensor, subnet_uid: int, hotkey: str) -> Optional[ModelMetadata]:
    """Retrieves model metadata on this subnet for specific hotkey"""
    metadata = bt.core.extrinsics.serving.get_metadata(subtensor, subnet_uid, hotkey)
    bt.logging.debug(f"metadata: {metadata}")
    
    if not metadata:
        return None
    
    try:
        commitment = metadata["info"]["fields"][0]
        bt.logging.debug(f"Commitment structure: {commitment}")
        
        if isinstance(commitment, dict):
            hex_data = commitment[list(commitment.keys())[0]][2:]
        elif isinstance(commitment, tuple) and hasattr(commitment, '__getitem__'):
            if len(commitment) > 0 and isinstance(commitment[0], dict) and 'Raw24' in commitment[0]:
                hex_data = commitment[0]['Raw24'][0][2:] if isinstance(commitment[0]['Raw24'][0], str) else ''.join([chr(c) for c in commitment[0]['Raw24'][0]])
            else:
                bt.logging.error(f"Unexpected commitment structure: {commitment}")
                return None
        else:
            bt.logging.error(f"Unsupported commitment type: {type(commitment)}")
            return None
        
        try:
            chain_str = bytes.fromhex(hex_data).decode() if isinstance(hex_data, str) else hex_data
        except Exception as e:
            bt.logging.error(f"Error decoding hex data: {e}")
            if isinstance(hex_data, tuple) and all(isinstance(i, int) for i in hex_data):
                chain_str = ''.join(chr(i) for i in hex_data)
            else:
                bt.logging.error(f"Unable to decode hex data: {hex_data}")
                return None
                
        model_id = None
        try:
            model_id = ModelId.from_compressed_str(chain_str)
        except Exception as e:
            bt.logging.error(
                f"Failed to parse the metadata on the chain for hotkey {hotkey}: {e}"
            )
            return None
        
        model_metadata = ModelMetadata(id=model_id, block=metadata["block"])
        return model_metadata
        
    except Exception as e:
        bt.logging.error(f"Error processing metadata: {e}")
        bt.logging.error(f"Stack trace:", exc_info=True)
        return None


def set_weights_with_err_msg(
        subtensor: bt.subtensor,
        wallet: bt.wallet,
        netuid: int,
        uids: [torch.LongTensor, list],
        weights: Union[torch.FloatTensor, list],

        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
) -> Tuple[bool, str, list[Exception]]:
    """Same as subtensor.set_weights, but with additional error messages."""
    uid = subtensor.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, netuid)
    retries = 0
    success = False
    message = "No attempt made. Perhaps it is too soon to set weights!"
    exceptions = []

    while (
            subtensor.blocks_since_last_update(netuid, uid) > subtensor.weights_rate_limit(netuid)  # type: ignore
            and retries < max_retries
    ):
        try:
            success, message = set_weights_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if (wait_for_inclusion or wait_for_finalization) and success:
                return success, message, exceptions

        except Exception as e:
            bt.logging.exception(f"Error setting weights: {e}")
            exceptions.append(e)
        finally:
            retries += 1

    return success, message, exceptions
