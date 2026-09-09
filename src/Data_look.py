import pandas as pd

import numpy as np


import matplotlib as plt
from dataset import load_dataset

dataset = load_dataset(
    "Dingdong-Inc/FreshRetailNet-50K",
    split="train[:100000]"
)

print(dataset)
print(dataset.column_names)

 