import pandas as pd
import torch
import torch.utils.data as data

from utils import get_vocabulary, get_labels


class Dataset(data.Dataset):
    def __init__(self, data: pd.DataFrame, vocabulary):
        super(Dataset, self).__init__()
        self.data = data
        self.vocabulary = vocabulary
        self.labels = get_labels(data)
        print("vocabulary size: ", len(self.vocabulary))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        row_tensor, label_tensor = self.row_iterator(row)
        return row_tensor, label_tensor

    def row_iterator(self, row) -> (list[int], int):
        row_text = row["text"].lower().split(" ")
        row_text = row_text[:1000]
        word_encoding_0 = torch.zeros((1000, len(self.vocabulary)))
        for id, word in enumerate(row_text):

            word_encoding_0[id][self.vocabulary.index(word)] = 1.0
        label_value = torch.tensor(self.labels[row["class_name"]])
        return word_encoding_0, label_value
