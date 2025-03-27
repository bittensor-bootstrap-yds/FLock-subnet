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
import math
import numpy as np
from FLockDataset import constants
from FLockDataset.utils.chain import assert_registered, read_chain_commitment
from FLockDataset.validator.chain import retrieve_model_metadata, set_weights_with_err_msg
from FLockDataset.validator.validator_utils import compute_score
from FLockDataset.validator.trainer import train_lora, download_dataset, clean_cache_folder
from FLockDataset.validator.database import ScoreDB  

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

        base_score = 1.0/255.0  # Default initial weight
        for uid in current_uids:
            self.score_db.insert_or_reset_uid(uid, hotkeys[uid], base_score)

        db_scores = self.score_db.get_scores(current_uids)
        self.weights = torch.tensor(db_scores, dtype=torch.float32)
        self.consensus = self.metagraph.C
        bt.logging.debug(f"Consensus: {self.consensus}")

        competition = read_chain_commitment(constants.SUBNET_OWNER_HOTKEY, self.subtensor, self.config.netuid)
        if competition is None:
            bt.logging.error("Failed to read competition commitment")
            return
        bt.logging.info(f"Competition commitment: {competition}")

        competitors = current_uids
        sample_size = min(self.config.miner_sample_size, len(competitors))
        uids_to_eval = self.rng.choice(competitors, sample_size, replace=False).tolist()
        bt.logging.debug(f"UIDs to evaluate: {uids_to_eval}")

        scores_per_uid = {uid: None for uid in uids_to_eval}
        metadata_per_uid = {uid: None for uid in uids_to_eval}
        block_per_uid = {uid: None for uid in uids_to_eval}
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


            duplicate_groups = []
            processed_uids = set()

            for uid_i, score_i in scores_per_uid.items():
                if score_i == 0 or uid_i in processed_uids:
                    continue
                    
                # Find all UIDs with nearly identical scores
                similar_uids = [uid_i]
                for uid_j, score_j in scores_per_uid.items():
                    if uid_i != uid_j and score_j != 0 and uid_j not in processed_uids:
                        if math.isclose(score_i, score_j, rel_tol=1e-9):
                            similar_uids.append(uid_j)
                
                # If we found duplicates, add them to a group
                if len(similar_uids) > 1:
                    duplicate_groups.append(similar_uids)
                    processed_uids.update(similar_uids)

            duplicates = set()
            for group in duplicate_groups:
                group.sort(key=lambda uid: block_per_uid[uid])
                
                for uid in group[1:]:
                    duplicates.add(uid)
                    scores_per_uid[uid] = 0  

            normalized_scores = {}
            for uid in uids_to_eval: 
                if scores_per_uid[uid] != 0: 
                    normalized_scores = compute_score(scores_per_uid[uid], competition.bench)
                else: 
                    normalized_scores[uid] = 0
            bt.logging.debug(f"Normalized scores: {normalized_scores}")

            new_weights = torch.zeros_like(self.weights)
            for uid, score in normalized_scores.items():
                new_weights[uid] = score
            
            for uid in uids_to_eval:
                if uid < len(new_weights):  
                    final_weight = new_weights[uid].item()
                    current_score = self.weights[uid].item()  
                    delta = final_weight - current_score
                    self.score_db.update_score(uid, delta)

            self.weights = new_weights
            bt.logging.debug(f'New weights: {new_weights}')
            bt.logging.debug(f'Consensus: {self.consensus}')

            set_weights_with_err_msg(
                subtensor=self.subtensor,
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=self.metagraph.uids,
                weights=new_weights,
            )

    async def run(self):
        while True:
            await self.run_step()

if __name__ == '__main__':
    asyncio.run(Validator().run())
