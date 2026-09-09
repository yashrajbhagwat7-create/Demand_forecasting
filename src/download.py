from datasets import load_dataset

dataset = load_dataset(
    "Dingdong-Inc/FreshRetailNet-50K"
)

print(dataset)
print(dataset["train"].column_names)