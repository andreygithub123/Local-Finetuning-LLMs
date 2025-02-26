import random
import string

def generate_password(length=12):
    """Generates a random password with uppercase, lowercase, digits, and symbols."""
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

# Example usage
print(generate_password())
