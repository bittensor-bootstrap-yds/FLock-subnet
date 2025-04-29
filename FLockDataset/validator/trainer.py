import os
import shutil
import torch
import yaml
from huggingface_hub import HfApi
from peft import LoraConfig
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from .dataset import SFTDataCollator, SFTDataset
from .constants import model2template
from FLockDataset import constants
import bittensor as bt

api = HfApi()


@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int


def download_dataset(
    namespace: str, revision: str, local_dir: str = "data", cache_dir: str = None
):
    # Create cache directory if it doesn't exist
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Set environment variable for this process
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "models")

    # Make sure local_dir is an absolute path
    if not os.path.isabs(local_dir):
        local_dir = os.path.abspath(local_dir)

    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    api.snapshot_download(
        repo_id=namespace, local_dir=local_dir, revision=revision, repo_type="dataset"
    )


def clean_cache_folder():
    try:
        shutil.rmtree("data")
    except:
        pass


def train_lora(
    lucky_num: int,
    cache_dir: str = None,
    data_dir: str = "data",
    eval_data_dir: str = "eval_data",
) -> float:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Set environment variable for this process
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "models")

    # set the same random seed to detect duplicate data sets
    from dotenv import load_dotenv

    load_dotenv()

    torch.manual_seed(lucky_num)
    torch.cuda.manual_seed(lucky_num)
    torch.cuda.manual_seed_all(lucky_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(lucky_num)
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(lucky_num)

    context_length = 512
    with open(f"FLockDataset/validator/training_args.yaml", "r") as f:
        all_training_args = yaml.safe_load(f)
    model_id = next(iter(all_training_args))
    training_args = LoraTrainingArguments(**all_training_args[model_id])

    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        save_strategy="no",
        output_dir=".",
        logging_dir=None,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        cache_dir=os.path.join(cache_dir, "models") if cache_dir else None,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
        cache_dir=os.path.join(cache_dir, "models") if cache_dir else None,
    )

    # Load dataset
    dataset = SFTDataset(
        file=os.path.join(data_dir, "data.jsonl"),
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    if len(dataset) > constants.EVAL_SIZE:
        bt.logging.info(
            f"Dataset has {len(dataset)} examples, expected {constants.EVAL_SIZE}, cheater detected"
        )
        return 9999999999999999

    eval_file_path = os.path.join(eval_data_dir, "eval_data.jsonl")
    if not os.path.exists(eval_file_path):
        # Look for any jsonl file in the eval directory
        jsonl_files = []
        for root, dirs, files in os.walk(eval_data_dir):
            for file in files:
                if file.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(root, file))

        if jsonl_files:
            # Use the first jsonl file found
            eval_file_path = jsonl_files[0]
            bt.logging.info(f"Using evaluation file: {eval_file_path}")
        else:
            bt.logging.error(f"No evaluation file found in {eval_data_dir}")
            return constants.BASELINE_LOSS

    eval_dataset = SFTDataset(
        file=eval_file_path,
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train()
    # Eval model
    eval_result = trainer.evaluate()

    return eval_result["eval_loss"]
