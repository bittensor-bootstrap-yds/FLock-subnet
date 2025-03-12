# FLockDataset: A Dataset Quality Competition Network for Machine Learning

FLockDataset (Bittensor Subnet) is a decentralized verification mechanism that incentivizes 
miners to create high-quality datasets for training machine learning models.


## Table of Contents
- [Compute Requirements](#compute-requirements)
- [Installation](#installation)
- [How to Run FLockDataset](#how-to-run-flockdataset)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [What is a Dataset Competition Network?](#what-is-a-dataset-competition-network)
  - [Role of a Miner](#role-of-a-miner)
  - [Role of a Validator](#role-of-a-validator)
- [Features of FLockDataset](#features-of-flockdataset)
  - [Hugging Face Integration](#hugging-face-integration)
  - [LoRA Training Evaluation](#lora-training-evaluation)

## Recommended Compute Requirements

For validators we recommend at least an [INSERT_EXPECTED_GPU] for efficient LoRA training evaluations, 
although an [INSERT_MINIMUM_GPU] could also be used. 

For miners, no specialized hardware is required as dataset creation and uploading is the primary task.

### Minimum Viable Compute Recommendations
- **For Validators:**
  - VRAM: [VRAM]
  - Storage: [STORAGE_NECESSARY]
  - RAM: [RAM]
  - CPU: [CPU]

- **For Miners:**
  - No specific GPU requirements
  - Hugging Face account with API access token
  - Storage: Some local storage for dataset creation and uploading [CHECK]

## Installation

### Overview
In order to run FLockDataset, you will need to install the FLockDataset package and set up a Hugging Face account. 
The following instructions apply to all major operating systems.

### Clone the repository
```bash
git clone https://github.com/organization/flockdataset.git
cd flockdataset
```

### Install dependencies
```bash
python3 -m pip install -e .
```

### Set up Hugging Face credentials
You'll need to create a Hugging Face account and generate an API token at https://huggingface.co/settings/tokens

Create a `.env` file in the project root and add your token:

```
HF_TOKEN=your_huggingface_token_here
```

You have now installed FLockDataset. You can now run a validator or a miner.

## How to Run FLockDataset

### Running a Miner

Before starting or registering your miner in FLockDataset, you'll need to:

// We should add instructions on how to do this 

1. Create a Hugging Face account and generate an API token
2. Create a Hugging Face repository to host your dataset
3. Prepare a high-quality dataset in JSON format

Once you have your dataset prepared, you can start your miner:

```bash
python3 neurons/miner.py --wallet.name [WALLET_NAME] --wallet.hotkey [WALLET_HOTKEY] --subtensor.network [NETWORK] --hf_repo_id [HF_REPO_ID] --netuid [NETUID] --dataset_path [PATH_TO_DATA_JSON] --logging.trace
```

Please replace the following with your specific configuration:
- `[WALLET_NAME]`
- `[WALLET_HOTKEY]`
- `[NETWORK]` (e.g., finney, local)
- `[HF_REPO_ID]` (your Hugging Face repository ID, e.g., username/repo-name)
- `[NETUID]` (the subnet UID)
- `[PATH_TO_DATA_JSON]` (path to your dataset file)

The miner script will:
1. Upload your dataset to Hugging Face
2. Register the dataset's location and commit ID // CHECK
3. Make it available for validators to evaluate

### Running a Validator

Validators train LoRA models on miners' datasets and evaluate their quality:

```bash
python3 neurons/validator.py --wallet.name [WALLET_NAME] --netuid [NETUID] --blocks_per_epoch [BLOCKS] --logging.trace
```

Please replace the following with your specific configuration:
- `[WALLET_NAME]`
- `[NETUID]` (the subnet UID)
- `[BLOCKS]` (number of blocks per epoch, default is 100) // we should get this from hyper params

## What is A Dataset Competition Network?

FLockDataset is a subnet where miners compete to create the highest quality datasets for training machine learning models. 
Validators evaluate these datasets by training LoRA models on them and comparing performance metrics.

### Role of a Miner

A miner is responsible for:
1. Creating high-quality datasets that effectively teach models useful capabilities
2. Formatting the data in a suitable structure (JSON)
3. Uploading datasets to Hugging Face
4. Registering dataset metadata on the blockchain // Document txn fee here 

Miners compete to create datasets that result in better-performing models when evaluated by validators.

### Role of a Validator

A validator is responsible for:
1. Retrieving miners' dataset metadata from the blockchain
2. Downloading datasets from Hugging Face
3. Training LoRA models using the datasets
4. Measuring model performance through evaluation loss
5. Computing win rates between datasets based on performance
6. Setting weights on the blockchain that determine miners' rewards

Validators ensure that miners are rewarded based on the actual quality of their datasets rather than computational resources.

## Features of FLockDataset

### Hugging Face Integration

FLockDataset leverages Hugging Face's infrastructure for dataset storage and version control:
- Miners maintain their own repositories for datasets
- Each submission is uniquely identifiable by repository + commit ID
- Version control is built-in through Hugging Face's Git-based system
- Validators can easily download and compare multiple datasets

### LoRA Training Evaluation

FLockDataset uses LoRA (Low-Rank Adaptation) for efficient evaluation:
- Trains models on each miner's dataset
- Evaluates performance on a standard test set
- Computes pairwise comparisons between datasets
- Distributes rewards based on relative performance
- Ensures fair competition regardless of dataset size

## How to Contribute

### Code Review

Project maintainers reserve the right to weigh the opinions of peer reviewers using common sense judgment and may also weigh based on merit. Reviewers that have demonstrated a deeper commitment and understanding of the project over time or who have clear domain expertise may naturally have more weight, as one would expect in all walks of life.

Where a patch set affects consensus-critical code, the bar will be much higher in terms of discussion and peer review requirements, keeping in mind that mistakes could be very costly to the wider community. This includes refactoring of consensus-critical code.

Where a patch set proposes to change the FLockDataset subnet, it must have been discussed extensively on the discord server and other channels, be accompanied by a widely discussed BIP and have a generally widely perceived technical consensus of being a worthwhile change based on the judgment of the maintainers.

That being said, we welcome all PRs for the betterment of the subnet and Bittensor as a whole. We are striving for improvement at every interval and believe through open communication and sharing of ideas will success be attainable.
