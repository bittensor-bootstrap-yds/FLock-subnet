# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import argparse
import asyncio
import torch
import typing
import bittensor as bt
import numpy as np
from FLockDataset import constants
from FLockDataset.utils.chain import assert_registered, read_chain_commitment
from FLockDataset.validator.chain import retrieve_model_metadata, set_weights_with_err_msg
from FLockDataset.validator.validator_utils import EvalQueue, compute_wins, adjust_for_vtrust
from FLockDataset.validator.trainer import train_lora, download_dataset, clean_cache_folder
from FLockDataset.validator.database import ScoreDB  # New database module

class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--blocks_per_epoch",
            type=int,
            default=360,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--miner_sample_size",
            type=int,
            default=3, 
            help="Number of miners to sample for each block.",
        )
        parser.add_argument(
            "--netuid", 
            type=int, 
            help="The subnet UID."
        )

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self):
        self.config = Validator.config()

        bt.logging(config=self.config)
        bt.logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)
        torch.backends.cudnn.benchmark = True
        self.uid = assert_registered(self.wallet, self.metagraph)
        self.weights = torch.zeros_like(torch.tensor(self.metagraph.S))
        self.uids_to_eval: typing.Dict[str, typing.List] = {} 
        self.score_db = ScoreDB("scores.db")
        self.rng = np.random.default_rng()    



    async def try_sync_metagraph(self) -> bool:
        bt.logging.trace("Syncing metagraph")
        try:
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            self.metagraph.save()
            bt.logging.info("Synced metagraph")
            return True
        except Exception as e:
            bt.logging.error(f"Error syncing metagraph: {e}")
            return False

    async def run_step(self):
        synced_metagraph = await self.try_sync_metagraph()
        if not synced_metagraph:
            bt.logging.warning("Failed to sync metagraph")
            return

        current_uids = self.metagraph.uids.tolist()
        hotkeys = self.metagraph.hotkeys
        for uid in current_uids:
            self.score_db.insert_or_reset_uid(uid, hotkeys[uid])

        # Update weights from consensus
        self.weights.copy_(torch.tensor(self.metagraph.C))
        self.consensus = self.metagraph.C
        if synced_metagraph:
            bt.logging.info("metagraph sync success: {}".format(self.consensus))
        else:
            bt.logging.warning("metagraph sync failed: {}".format(self.consensus))

        # get competition info
        competition = read_chain_commitment(constants.SUBNET_OWNER_HOTKEY, self.subtensor, self.config.netuid)
        if competition is None:
            bt.logging.error("Failed to read competition commitment")
            return
        bt.logging.info(f"Competition commitment: {competition}")

        competitors = current_uids
        sample_size = min(self.config.miner_sample_size, len(competitors))
        uids_to_eval = self.rng.choice(competitors, sample_size, replace=False).tolist()
        bt.logging.debug(f"UIDs to evaluate: {uids_to_eval}")

        hotkeys_to_eval = [self.metagraph.hotkeys[uid] for uid in uids_to_eval]
        scores_per_uid = {uid: None for uid in uids_to_eval}
        metadata_per_uid = {uid: None for uid in uids_to_eval}
        block_per_uid = {uid: None for uid in uids_to_eval}
        is_duplicate = []

        lucky_num = int.from_bytes(os.urandom(4), 'little')

        for uid in uids_to_eval: 
            metadata = retrieve_model_metadata(self.subtensor, self.config.netuid, self.metagraph.hotkeys[uid])
            if metadata is not None: 
                try: 
                    download_dataset(metadata.id.namespace, metadata.id.commit)
                    download_dataset(constants.eval_namespace, constants.eval_commit, local_dir="eval_data")
                    eval_loss = train_lora(lucky_num)

                    metadata_per_uid[uid] = metadata
                    scores_per_uid[uid] = eval_loss
                    block_per_uid[uid] = metadata.block
                except Exception as e:
                    bt.logging.error(f"train error: {e}")
                    scores_per_uid[uid] = 0
                finally:
                    clean_cache_folder()
            else:
                scores_per_uid[uid] = 0


        is_duplicate = []
        for i, uid_i in enumerate(uids_to_eval):
            if scores_per_uid[uid_i] == 0 or uid_i in is_duplicate:
                continue
                
            for j, uid_j in enumerate(uids_to_eval[i+1:], i+1):
                if scores_per_uid[uid_j] == 0 or uid_j in is_duplicate:
                    continue
                    
                # Check if scores are nearly identical
                if math.isclose(scores_per_uid[uid_i], scores_per_uid[uid_j], rel_tol=1e-9):
                    # Determine which is the duplicate based on block number
                    if block_per_uid[uid_i] > block_per_uid[uid_j]:
                        is_duplicate.append(uid_j)
                    else:
                        is_duplicate.append(uid_i)
                        break  # No need to check further for uid_i







#             consensus_map = {uid: self.weights[uid].item() for uid in consensus}
#             bt.logging.info(
#                 f"Consensus for competition {competition.competition_id}: {consensus_map}"
#             )

# 
#             # Sync the first few models, we can sync the rest while running.
# 
#             uids_to_sync = list(uids_to_eval)[:self.config.miner_sample_size]
#             hotkeys = [self.metagraph.hotkeys[uid] for uid in uids_to_sync]
# 
#             print(f"hotkeys: {hotkeys}")
#             scores_per_uid = {uid: None for uid in uids_to_sync}
#             metadata_per_uid = {uid: None for uid in uids_to_sync}
#             block_per_uid = {uid: None for uid in uids_to_sync}
#             is_duplicate = []
#             lucky_num = int.from_bytes(os.urandom(4), 'little')
#             for uid in uids_to_sync:
#                 metadata = retrieve_model_metadata(self.subtensor, self.config.netuid, self.metagraph.hotkeys[uid])
#                 print(f"uid: {uid}")
#                 print(f"results: {metadata}")
#                 if metadata is not None:
#                     try:
#                         download_dataset(metadata.id.namespace, metadata.id.commit)
#                         download_dataset(constants.eval_namespace, constants.eval_commit, local_dir="eval_data")
#                         eval_loss = train_lora(lucky_num)
#                         other_uid = [k for k, v in scores_per_uid.items() if
#                                      v is not None and math.isclose(eval_loss, v, rel_tol=1e-9)]
# 
#                         # check if duplicate
#                         for range_id in other_uid:
#                             if metadata_per_uid[range_id].block > metadata.block:
#                                 is_duplicate.append(range_id)
#                             else:
#                                 is_duplicate.append(uid)
# 
#                         metadata_per_uid[uid] = metadata
#                         scores_per_uid[uid] = eval_loss
#                         block_per_uid[uid] = metadata.block
#                     except Exception as e:
#                         bt.logging.error(f"train error: {e}")
#                         scores_per_uid[uid] = 0
#                     finally:
#                         clean_cache_folder()
#                 else:
#                     scores_per_uid[uid] = 0
#             uids_whitelist = [item for item in uids_to_sync if item not in is_duplicate]
#             wins, win_rate = compute_wins(
#                 uids_whitelist, scores_per_uid,
#                 block_per_uid
#             )
#             model_weights = torch.tensor(
#                 [win_rate[uid] for uid in win_rate.keys()], dtype=torch.float32
#             )
#             step_weights = torch.softmax(model_weights / constants.temperature, dim=0)
#             new_weights = torch.zeros_like(self.weights)
#             for i, uid_i in enumerate(win_rate.keys()):
#                 new_weights[uid_i] = step_weights[i]
# 
#             consensus_alpha = constants.CONSTANT_ALPHA
#             lr = constants.lr
#             self.weights = (
#                     (1 - lr) * self.weights + lr * new_weights
#             )
#             # To prevent the weights from completely diverging from consensus, blend in the consensus weights.
#             C_normalized = torch.tensor(self.consensus / self.consensus.sum()).nan_to_num(0.0)
#             self.weights = (1 - consensus_alpha) * self.weights + consensus_alpha * C_normalized
#             self.weights = self.weights.nan_to_num(0.0)
#             adjusted_weights = adjust_for_vtrust(self.weights.cpu().numpy(), self.consensus)
#             adjusted_weights = torch.tensor(adjusted_weights, dtype=torch.float32)
# 
#             bt.logging.debug(f'new weights: {new_weights}')
#             bt.logging.debug(f'self weights: {self.weights}')
#             bt.logging.debug(f'consensus: {self.consensus}')
#             bt.logging.debug(f'adjusted_weights: {adjusted_weights}')
# 
#             set_weights_with_err_msg(
#                 subtensor=self.subtensor,
#                 wallet=self.wallet,
#                 netuid=self.config.netuid,
#                 uids=self.metagraph.uids,
#                 weights=adjusted_weights,
#             )


    async def run(self):
        while True:
            await self.run_step()

if __name__ == '__main__':
    asyncio.run(Validator().run())
