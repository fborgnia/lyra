import json
import random
from pathlib import Path
from faker import Faker

# Initialize Faker to generate synthetic data
fake = Faker()

# Define a list of templates for generating context-question-answer triplets.
# Each template contains placeholders that will be filled with fake data.
TEMPLATES = [
    {
        "context": "My name is {name}.",
        "question": "What is my name?",
        "answer": "Your name is {name}.",
        "data": lambda: {"name": fake.name()},
    },
    {
        "context": "I live in {city}.",
        "question": "Where do I live?",
        "answer": "You live in {city}.",
        "data": lambda: {"city": fake.city()},
    },
    {
        "context": "My favorite color is {color}.",
        "question": "What is my favorite color?",
        "answer": "Your favorite color is {color}.",
        "data": lambda: {"color": fake.color_name()},
    },
    {
        "context": "I work as a {job}.",
        "question": "What is my job?",
        "answer": "You work as a {job}.",
        "data": lambda: {"job": fake.job()},
    },
    {
        "context": "The secret code is {random_number}.",
        "question": "What is the secret code?",
        "answer": "The secret code is {random_number}.",
        "data": lambda: {"random_number": fake.random_number(digits=5)},
    },
    {
        "context": "My pet's name is {first_name}.",
        "question": "What is my pet's name?",
        "answer": "Your pet's name is {first_name}.",
        "data": lambda: {"first_name": fake.first_name()},
    },
    {
        "context": "I am traveling to {country}.",
        "question": "Where am I traveling to?",
        "answer": "You are traveling to {country}.",
        "data": lambda: {"country": fake.country()},
    },
]

def generate_dataset(num_samples: int) -> list[dict]:
    """
    Generates a dataset of synthetic conversations for semantic alignment training.

    Args:
        num_samples: The number of conversation triplets to generate.

    Returns:
        A list of dictionaries, where each dictionary represents a single
        (context, question, answer) triplet.
    """
    dataset = []
    for _ in range(num_samples):
        template = random.choice(TEMPLATES)
        
        # Generate the specific fake data required by the chosen template
        template_data = template["data"]()
        
        # Create the conversation triplet by formatting the template strings
        # with the generated fake data.
        dataset.append(
            {
                "context": template["context"].format(**template_data),
                "user_input": template["question"].format(**template_data),
                "answer": template["answer"].format(**template_data),
            }
        )
    return dataset

def main():
    """
    Main function to generate the dataset and save it to a JSON file.
    """
    num_samples = 500
    print(f"Generating {num_samples} samples for the semantic alignment dataset...")

    # Generate the dataset
    dataset = generate_dataset(num_samples)

    # Define the output path and ensure the directory exists
    output_path = Path(__file__).parent.parent / "data" / "semantic_alignment.json"
    output_path.parent.mkdir(exist_ok=True)

    # Write the dataset to the JSON file
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Successfully generated dataset at: {output_path}")

if __name__ == "__main__":
    main()

