import torch
from transformers import AutoTokenizer
import yaml
from pathlib import Path

# Load YAML config to get tokenizer path
with open("super_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load tokenizer
tokenizer_dir = config["paths_config"]["tokenizer_path"]  # Path to tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# Path to the tokenized dataset (Update this if needed)
tokenized_file_path = Path("./tokenized_datasets/3_conversation_examples.pt")  # Change to your actual .pt file

# Load tokenized data
tokenized_data = torch.load(tokenized_file_path)

# Extract tokens (Assuming `input_ids` is inside the saved object)
if "input_ids" in tokenized_data:
    token_ids = tokenized_data["input_ids"]  # Extract token IDs tensor
else:
    token_ids = tokenized_data  # If directly saved

# Convert tensor to list if needed
if isinstance(token_ids, torch.Tensor):
    token_ids = token_ids.tolist()

# Decode tokens back into human-readable text
decoded_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)

# Print the detokenized content
for i, text in enumerate(decoded_text):
    print(f"Decoded Sample {i+1}:")
    print(text)
    print("=" * 50)  # Separator for readability
