import random
import string

def generate_password(length=12):
    """Generates a random password with uppercase, lowercase, digits, and symbols."""
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

# Example usage
print(generate_password())


import random

def random_func():
    return random.randint(1, 100)

def update_python_file():
    file_path = "generated_code.py"  # Change this to your target .py file

    try:
        code = get_code_message().strip()  # Ensure no extra whitespace
        print("Retrieved Code:\n", code)

        # Open the file in append mode and write the new code
        with open(file_path, "a", encoding="utf-8") as file:
            file.write("\n\n")  # Ensure separation between previous and new code
            file.write(code)
            file.write("\n")  # Newline for better readability

        print(f"Updated {file_path} successfully.")

    except Exception as e:
        print(f"Error updating {file_path}: {e}")
