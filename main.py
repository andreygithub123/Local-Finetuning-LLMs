import torch
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AdamW,
    get_scheduler,
)
from accelerate import load_checkpoint_and_dispatch
from pathlib import Path
import yaml
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

with open("super_config.yaml","r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Extract paths from YAML
tokenized_folder = Path(config["paths_config"]["tokenized_datasets"])  # Tokenized dataset folder
tokenizer_dir = Path(config["paths_config"]["tokenizer_path"])  # Tokenizer model path
output_dir = Path(config["output_config"]["output_dir"])  # Checkpoints directory
logging_dir = Path(config["output_config"]["logging_dir"])  # Logs directory
model_save_path = Path(config["output_config"]["model_save_path"])  # Final model saving path

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, tokens):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],  # Labels for causal LM
        }

if __name__ == "__main__":
      # Find tokenized dataset
    tokenized_files = list(tokenized_folder.glob("*.pt"))
    
    if not tokenized_files:
        logging.error("No tokenized datasets found in tokenized_datasets folder.")
        raise FileNotFoundError("Tokenized dataset is missing.")
    
    # Load the first available tokenized dataset
    tokenized_file = tokenized_files[0]

    # Load the pre-tokenized dataset
    try:
        tokenized_data = torch.load(tokenized_file)
        logging.info("Tokenized dataset successfully loaded.")
    except Exception as e:
        logging.error("Failed to load tokenized dataset.", exc_info=True)
        raise e

    # Train-test split
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    train_indices, val_indices = train_test_split(range(len(input_ids)), test_size=0.2, random_state=42)
    train_tokens = {"input_ids": input_ids[train_indices], "attention_mask": attention_mask[train_indices]}
    val_tokens = {"input_ids": input_ids[val_indices], "attention_mask": attention_mask[val_indices]}
    train_dataset = CustomDataset(train_tokens)
    val_dataset = CustomDataset(val_tokens)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
       
        # Load model with device mapping
        model = AutoModelForCausalLM.from_pretrained(
            tokenizer_dir,
            device_map="auto",  # Automatically distribute layers
            max_memory={0: "24GB", "cpu": "128GB"},  # Adjust memory usage
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            offload_folder="./offload",  # Folder for offloading to disk
        )
        logging.info("Model successfully loaded with CPU offloading.")
    except Exception as e:
        logging.error("Error occurred while loading the model.", exc_info=True)
        raise e

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        logging_dir=str(logging_dir),
        logging_steps=1,
        eval_steps=10,
        save_steps=100,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        bf16=True,  
        report_to="wandb",
        dataloader_pin_memory=True,
    )

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    logging.info(f"Fine-tuned model saved at: {model_save_path}")