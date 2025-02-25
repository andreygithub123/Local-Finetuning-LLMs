# Local-Finetuning-LLMs
This repository displayes the strategy that I used in order to finetune a MISTRAL-7B using a single RTX 4090 Windforce V2 24GB VRAM. This is no way best practices in production, but at least you can get a glimpse of testing your training hypothesis. The pipeline can work with other hyperparameter settings ( thus it make no sense for following this pipeline ).

## Finetune guide ( on a virtual env)

1. Install dependencies: 
````bash
pip install -r requirements.txt ( it is recommended to use python 3.10 )
````
2. Put shareGPT instruct-data into `training_datasets/` folder
3. Put `.safetensors` LLM into `local_model` folder
4. Run the main pipeline
````bash
python main.py
or
python3 main.py
````
5. Finetuned model will be in `finetuned_model` folder in `.safetensors` format


## Workflow

`Preprocessing data` :
1. The script takes a ShareGPT-like format conversation from a `.jsonl` file and transforms it into ChatML format.
2. The chatML conversation is then tokenized and cached into '.pt' format ( in order to reduce the preprocessing step from finetuning pipeline)

### Example of ShareGPT Data (JSON)
```json
{
  "conversation": [
    {
      "role": "system",
      "content": "You are a helpful cybersecurity assistant."
    },
    {
      "role": "user",
      "content": "What should the organization do to formulate an information security risk treatment plan?"
    },
    {
      "role": "assistant",
      "content": "The organization should identify and evaluate risks, formulate an information security risk treatment plan, and obtain risk owners' approval of the plan and acceptance of the residual risks."
    }
  ]
}
```
### Example of chatML converted data
```
|im_start|>system
You are a helpful cybersecurity assistant.
<|im_end|>
<|im_start|>user
What should the organization do to formulate an information security risk treatment plan?
<|im_end|>
<|im_start|>assistant
The organization should identify and evaluate risks, formulate an information security risk treatment plan, and obtain risk owners' approval of the plan and acceptance of the residual risks.
<|im_end|>
```

## Hyperparameter settings
It is recommended to run with a batch size of 1 and increase the gradient_accumulation_step as needed. In this way we will mitigate the risk of running out of VRAM. I would personally recommend to not change the optimizer and scheduler but you can play with it as you want.

### Training Arguments
- **Batch Size**: The script uses `per_device_train_batch_size=1` and `gradient_accumulation_steps=4` to help mitigate VRAM limitations while maintaining an effective batch size.
- **Epochs**: Training runs for `5` epochs (`num_train_epochs=5`).
- **Evaluation & Logging**:
  - The model is evaluated every `10` steps (`eval_steps=10`).
  - Logs are generated every `1` step (`logging_steps=1`).
  - Checkpoints are saved every `100` steps (`save_steps=100`) with a limit of `2` saved models (`save_total_limit=2`).
- **Mixed Precision Training**: `bf16=True` enables **bfloat16** precision, optimizing memory usage without significant loss of accuracy.
- **Memory Management**: `dataloader_pin_memory=True` improves data loading efficiency.
- **Experiment Tracking**: The training process is logged with **Weights & Biases** (`report_to="wandb"`).

### Optimizer & Scheduler
- **Optimizer**: The model is trained using the `AdamW` optimizer with:
  - Learning rate: `1e-5`
  - Weight decay: `0.01`
- **Learning Rate Scheduler**: A **linear scheduler** is used, with the number of training steps computed dynamically based on dataset size and training epochs.

### Recommendations
- **Batch Size & Gradient Accumulation**: It is recommended to use a small batch size (`1`) and adjust `gradient_accumulation_steps` as needed to prevent VRAM overflow.
- **Optimizer & Scheduler**: The current settings (`AdamW` + **linear scheduler**) provide stable performance. However, you can experiment with different optimizers and schedulers based on your use case.

## YAML Configuration File

This configuration file defines essential paths, output directories, and the execution pipeline for the training and tokenization process.

### Paths Configuration
The `paths_config` section specifies the directories used throughout the data processing and training pipeline:

- **Dataset Folder**: `./training_datasets`  
  Location where the raw `.jsonl` datasets are stored.
- **Output Folder**: `./chatml_datasets`  
  Directory where converted **ChatML** formatted files will be saved.
- **ChatML Datasets**: `./chatml_datasets`  
  Used again during tokenization.
- **Tokenized Datasets**: `./tokenized_datasets`  
  Folder where tokenized datasets will be stored.
- **Tokenizer Path**: `./local_model`  
  Path to the tokenizer model used for tokenization.

### Output Configuration
The `output_config` section defines where the processed model artifacts will be stored:

- **Fine-tuned Model Path**: `./finetuned_model`  
  The directory where the fine-tuned model and tokenizer will be saved.
- **Checkpoint Directory**: `./checkpoints_model`  
  Folder where model checkpoints will be stored. This also serves as the **run name on Weights & Biases (WandB)**.
- **Logging Directory**: `./logs_model`  
  Location for **TensorBoard logs**.

### Pipeline Execution Order
The `pipeline_order` section defines the sequence of scripts that will be executed:

1️. **Convert dataset to ChatML format** (`convert_to_chatml.py`)  
2️. **Tokenize and cache the dataset** (`chatml_cache_tokenzier.py`)  
3️. **Train the model** (`main.py`)  

## Customization
You can modify the paths and execution order in `config.yaml` based on your project structure and specific needs.

---

## Resources

Check the Jupyter Notebook for a detailed explanation of the fine-tuning process:  [View Notebook on Google Colab](https://colab.research.google.com/drive/1iire0j9Fz-BVngFJlcprdmqQ89uR7QDX?usp=sharing)


