import pandas as pd
import torch
import torch.utils.data as data

from utils import get_vocabulary, get_labels

#
#
# class Dataset(data.Dataset):
#     def __init__(self, data: pd.DataFrame, vocabulary, context_len=500):
#         super(Dataset, self).__init__()
#         self.data = data
#         self.vocabulary = vocabulary
#         self.labels = get_labels(data)
#         self.context_len = context_len
#         print("vocabulary size: ", len(self.vocabulary))
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         row_tensor, label_tensor = self.row_iterator(row)
#         return row_tensor, label_tensor
#
#     def row_iterator(self, row) -> (list[int], int):
#         row_text = row["text"].lower().split(" ")
#         row_text = row_text[: self.context_len]
#         word_encoding_0 = torch.zeros((self.context_len, len(self.vocabulary)))
#         for id, word in enumerate(row_text):
#             word_encoding_0[id][self.vocabulary.index(word)] = 1.0
#         label_value = torch.tensor(self.labels[row["class_name"]])
#         return word_encoding_0, label_value


class Dataset(data.Dataset):
    """
    Custom PyTorch dataset for processing text data and generating one-hot encodings.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing text and class labels.
    vocabulary : list
        A list of unique words used to create one-hot encodings.
    context_len : int, optional
        Maximum length of the input text to be considered, by default 500.

    Attributes
    ----------
    data : pd.DataFrame
        The input data containing text and labels.
    vocabulary : list
        List of unique words for encoding.
    labels : dict
        Mapping of class names to numeric labels, retrieved using `get_labels`.
    context_len : int
        Maximum length of text sequences for encoding.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx)
        Retrieves the input and label tensors for the given index.
    row_iterator(row)
        Processes a single row to generate the one-hot encoded tensor and label.

    Examples
    --------
    >>> from utils import get_vocabulary
    >>> data = pd.DataFrame({
    >>>     "text": ["example text one", "another example"],
    >>>     "class_name": ["class1", "class2"]
    >>> })
    >>> vocabulary = get_vocabulary(data)
    >>> dataset = Dataset(data, vocabulary)
    >>> len(dataset)
    2
    >>> input_tensor, label_tensor = dataset[0]
    """

    def __init__(self, data: pd.DataFrame, vocabulary, context_len=500):
        super(Dataset, self).__init__()
        self.data = data
        self.vocabulary = vocabulary
        self.labels = get_labels(data)
        self.context_len = context_len
        self.words_idx = {word: idx for idx, word in enumerate(vocabulary)}
        print("vocabulary size: ", len(self.vocabulary))

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the input and label tensors for the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            - One-hot encoded tensor of shape (context_len, len(vocabulary)).
            - Corresponding label tensor.
        """
        row = self.data.iloc[idx]
        row_tensor, label_tensor = self.row_iterator(row)
        return row_tensor, label_tensor

    def row_iterator(self, row):
        """
        Processes a single row to generate the one-hot encoded tensor and label.

        Parameters
        ----------
        row : pd.Series
            A row from the DataFrame containing the text and class name.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            - One-hot encoded tensor of shape (context_len, len(vocabulary)).
            - Label tensor as an integer.
        """
        row_text = row["text"].lower().split(" ")
        row_text = row_text[: self.context_len]
        word_indices = torch.zeros(self.context_len, dtype=torch.long)
        for id, word in enumerate(row_text):
            if word in self.words_idx:
                word_indices[id] = self.words_idx[word]

        label_value = self.labels[row["class_name"]]
        return word_indices, label_value
