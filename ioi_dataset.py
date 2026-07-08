import random
import json
import os

class IOIDataset:
    """
    To generate and manage IOI (Indirect Object Identification) datasets.
    """

    NAMES = [
        "John",
        "Mary",
        "Alice",
        "Bob",
        "Tom",
        "Sarah"
    ]

    OBJECTS = [
        "a bottle of milk",
        "a book",
        "a gift",
        "a letter",
        "a phone"
    ]

    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.dataset = []

    def generate_sentence(self):
        """
        Generates a single sentence with a prompt, correct answer, and incorrect answer.
        ABB format.
        """
        name1, name2 = random.sample(self.NAMES, 2)
        obj = random.choice(self.OBJECTS)
        prompt = (
            f"When {name1} and {name2} went to the store, "
            f"{name1} gave {obj} to"
        )

        return {
            "prompt": prompt,
            "correct": name2,
            "incorrect": name1
        }

    def generate_dataset(self):
        """
        Generates complete dataset.
        """
        self.dataset = [self.generate_sentence() for _ in range(self.num_samples)]

    def save_dataset(self, file_path):
        """
        Saves the generated dataset to a JSON file.

        Args:
            file_path (str): The path to save the dataset.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.dataset, f, indent=4)
        print(f"Dataset saved to {file_path}")

    def load_dataset(self, file_path):
        """
        Loads a dataset from a JSON file.

        Args:
            file_path (str): The path to load the dataset from.
        """
        with open(file_path, "r") as f:
            self.dataset = json.load(f)
        print(f"Dataset loaded from {file_path}")

    def display_dataset(self):
        for item in self.dataset:
            print("----------------")
            print("Prompt:", item["prompt"])
            print("Correct:", item["correct"])
            print("Incorrect:", item["incorrect"])

def main():
    num_samples = 5
    folder_path = "dataset"
    file_name = "ioi_dataset.json"
    file_path = os.path.join(folder_path, file_name)
    ioi = IOIDataset(num_samples)
    ioi.generate_dataset()
    ioi.display_dataset()
    ioi.save_dataset(file_path)
    
if __name__ == "__main__":
    main()