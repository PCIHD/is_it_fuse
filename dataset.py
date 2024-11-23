import pandas as pd
import torch
import torch.utils.data as data

from utils import get_vocabulary, get_labels


class Dataset(data.Dataset):
    def __init__(self, data: pd.DataFrame):
        super(Dataset, self).__init__()
        self.data = data
        self.vocabulary = get_vocabulary(data)
        self.labels = get_labels(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        row_tensor, label_tensor = self.row_iterator(row)
        return torch.tensor(row_tensor), torch.tensor(label_tensor)

    def row_iterator(self, row) -> (list[int], int):
        row_text = row["text"].lower().split(" ")
        row_tensor = []
        for word in row_text:
            word_encoding_0 = [0.0 for _ in range(len(self.vocabulary))]
            word_encoding_0[self.vocabulary.index(word)] = 1.0
            row_tensor.append(word_encoding_0)
        label_value = self.label_encoding[row["class_name"]]
        return row_tensor, label_value
