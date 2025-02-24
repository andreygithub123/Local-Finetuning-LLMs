print("START FINETUNING PROCESS.")

# Orchestrate and execute pipelines according to the super-config in order.
import subprocess
import os
import yaml
import sys
from pathlib import Path

with open("super_config.yaml", "r") as f:
    super_config = yaml.safe_load(f)


def run_processing_script(script_name, project_root):
    """
    Runs a processing script (e.g., "processing.py") from the project root.

    :param script_name: The name of the script to execute.
    :param project_root: Root directory of the project.
    """
    script_path = project_root / script_name  # Locate script in project root

    if not script_path.is_file():
        print(f"Error: {script_path} not found.")
        return  # Skip execution

    # Run the script without changing directories
    subprocess.run([sys.executable, str(script_path)], check=True)

def main():
    project_root = Path(__file__).parent.resolve()  # Get project root directory
    pipeline_steps = super_config["pipeline_order"]

    for step in pipeline_steps:
        script_name = step["script"]  
        run_processing_script(script_name, project_root)

if __name__ == "__main__":
    main()
