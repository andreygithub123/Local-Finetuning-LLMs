import json
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load YAML config
with open("super_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

dataset_folder = Path(config["paths_config"]["dataset_folder"])  # Directory containing .jsonl files
output_folder = Path(config["paths_config"]["output_folder"])  # Where .txt files will be saved
output_folder.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists


def check_format(data, file_name, line_number):
    """
    Validates the format of a JSON conversation entry.
    
    :param data: A parsed JSON object.
    :param file_name: Name of the JSONL file being checked.
    :param line_number: Line number where the issue occurs.
    :return: True if the format is correct, False otherwise.
    """
    if not isinstance(data, dict):
        logging.error(f"[{file_name} - Line {line_number}] Invalid JSON format: Root element should be a dictionary.")
        return False

    if "conversation" not in data:
        logging.error(f"[{file_name} - Line {line_number}] Missing 'conversation' key in JSON object.")
        return False

    conversation = data["conversation"]
    
    if not isinstance(conversation, list):
        logging.error(f"[{file_name} - Line {line_number}] 'conversation' should be a list.")
        return False

    if len(conversation) < 2:  # At least system + one user message
        logging.error(f"[{file_name} - Line {line_number}] Invalid conversation: Must contain at least a system message and one user interaction.")
        return False

    # Ensure the first message is from the "system"
    first_message = conversation[0]
    if not isinstance(first_message, dict) or first_message.get("role") != "system":
        logging.error(f"[{file_name} - Line {line_number}] First conversation message must be from 'system'.")
        return False

    # Check each message in the conversation
    for msg in conversation:
        if not isinstance(msg, dict):
            logging.error(f"[{file_name} - Line {line_number}] Invalid format: Each message should be a dictionary.")
            return False

        if "role" not in msg or "content" not in msg:
            logging.error(f"[{file_name} - Line {line_number}] Each message must contain 'role' and 'content'.")
            return False

        if msg["role"] not in {"system", "user", "assistant"}:
            logging.error(f"[{file_name} - Line {line_number}] Invalid role: {msg['role']} (must be 'system', 'user', or 'assistant').")
            return False

        if not isinstance(msg["content"], str) or not msg["content"].strip():
            logging.error(f"[{file_name} - Line {line_number}] Invalid content: 'content' must be a non-empty string.")
            return False

    return True


def convert_to_chatml_format(input_jsonl, output_txt):
    """
    Convert a JSONL dataset into ChatML format and save it as a text file.
    """
    try:
        logging.info(f"Processing {input_jsonl}...")

        with open(input_jsonl, 'r', encoding="utf-8") as infile, open(output_txt, 'w', encoding="utf-8") as outfile:
            for line_number, line in enumerate(infile, start=1):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logging.error(f"[{input_jsonl.name} - Line {line_number}] JSON decoding error. Skipping entry.")
                    continue

                if not check_format(data, input_jsonl.name, line_number):
                    logging.error(f"[{input_jsonl.name} - Line {line_number}] Skipping invalid entry.")
                    continue

                conversation = data["conversation"]

                # Start with the system message
                chatml_data = "<|im_start|>system\n"
                chatml_data += f"{conversation[0]['content']}\n<|im_end|>\n"

                # Add user and assistant turns
                for message in conversation[1:]:
                    role = message["role"]  # "user" or "assistant"
                    content = message["content"]
                    chatml_data += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

                # Write to the output file
                outfile.write(chatml_data.strip() + "\n\n")

        logging.info(f"ChatML dataset saved to {output_txt}.")
    except Exception as e:
        logging.error(f"[{input_jsonl.name}] Failed to convert to ChatML format.", exc_info=True)
        raise e


# Find all .jsonl files in dataset folder
jsonl_files = list(dataset_folder.glob("*.jsonl"))

if not jsonl_files:
    logging.warning("No JSONL files found in the dataset folder.")
else:
    for jsonl_file in jsonl_files:
        output_txt = output_folder / f"{jsonl_file.stem}.txt"  # Save with the same name but .txt extension
        convert_to_chatml_format(jsonl_file, output_txt)
