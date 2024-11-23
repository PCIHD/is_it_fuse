import torch
import lightning as L
import pandas as pd
from torch.utils.data import TensorDataset

from dataset import Dataset
from utils import vectorize_text

df = pd.read_parquet("./data_foundation/curated/curated_dataset.parquet")
inputs, labels = vectorize_text(df)
inputs = torch.Tensor(inputs)
labels = torch.Tensor(labels)
dataset = Dataset(df)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
