from datasets import load_from_disk
import pandas as pd

# Load GoEmotions dataset
dataset = load_from_disk("go_emotions_dataset")

# Use 'train' split for training, 'test' for evaluation
train_data = dataset["train"]
test_data = dataset["test"]

print(train_data[0])  # sample data