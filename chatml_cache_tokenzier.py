import torch
from transformers import AutoTokenizer
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load YAML config
with open("super_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Paths from YAML
chatml_folder = Path(config["paths_config"]["chatml_datasets"])  # Input folder for ChatML datasets
tokenized_folder = Path(config["paths_config"]["tokenized_datasets"])  # Output folder for tokenized datasets
tokenizer_dir = Path(config["paths_config"]["tokenizer_path"])  # Tokenizer model path

# Ensure output folder exists
tokenized_folder.mkdir(parents=True, exist_ok=True)

def load_and_tokenize_chatml_dataset(file_path, tokenizer, max_length=2048, padding="longest"):
    """
    Load and tokenize a ChatML dataset with padding and truncation.
    """
    try:
        logging.info(f"Loading dataset from {file_path}...")

        # Read the ChatML dataset
        with open(file_path, "r", encoding="utf-8") as f:
            conversations = f.read().split("\n\n")  # Each conversation is separated by two newlines

        # Tokenize conversations with padding and truncation
        logging.info(f"Tokenizing {len(conversations)} conversations...")
        tokens = tokenizer(
            conversations,
            max_length=max_length,
            truncation=False,
            padding=padding,  # Add padding
            return_tensors="pt"
        )
        logging.info(f"Tokenization complete for {file_path}.")
        return tokens
    except Exception as e:
        logging.error(f"Error loading or tokenizing dataset: {file_path}", exc_info=True)
        raise e


if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.padding_side = "right"

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Padding token set to EOS token: {tokenizer.pad_token}")

    # Find all .txt ChatML datasets in chatml_datasets/
    chatml_files = list(chatml_folder.glob("*.txt"))

    if not chatml_files:
        logging.warning("No ChatML dataset files found in chatml_datasets folder.")
    else:
        for chatml_file in chatml_files:
            try:
                # Tokenize dataset
                tokenized_data = load_and_tokenize_chatml_dataset(
                    file_path=chatml_file,
                    tokenizer=tokenizer,
                    max_length=1024,  # Truncate sequences longer than 2048 tokens
                    padding="longest"  # Pad sequences to the length of the longest sequence
                )

                # Define save path
                save_path = tokenized_folder / f"{chatml_file.stem}.pt"

                # Save tokenized dataset
                torch.save(tokenized_data, save_path)
                logging.info(f"Tokenized dataset saved to {save_path}.")
            except Exception as e:
                logging.error(f"Failed to tokenize and save the dataset: {chatml_file}", exc_info=True)
