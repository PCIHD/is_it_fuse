import pandas as pd
import re

import torch


def vectorize_text(text: pd.DataFrame):
    vocabulary = get_vocabulary(text)
    word_encoding = [0.0 for _ in range(len(vocabulary))]
    classes = text["class_name"].unique()
    label_encoding = get_labels(classes)
    inputs = []
    labels = []
    return_values = text.apply(
        row_iterator, axis=1, args=(word_encoding, label_encoding, vocabulary)
    )
    # for index, row in text.iterrows():
    #     input_value, label_value = row_iterator(row, word_encoding, label_encoding, vocabulary)
    #     inputs.append(input_value)
    #     labels.append(label_value)
    return return_values, vocabulary


def get_labels(classes):
    unique_classes = classes["class_name"].unique()
    label_encoding = {}
    for i, encoding_name in enumerate(unique_classes):
        encoding = torch.zeros(len(unique_classes))
        encoding[i] = 1.0
        label_encoding[encoding_name] = encoding
    return label_encoding


def row_iterator(row, word_encoding, label_encoding, vocabulary) -> (list[int], int):
    row_text = row["text"].lower().split(" ")
    row_tensor = []
    for word in row_text:
        word_encoding_0 = word_encoding.copy()
        word_encoding_0[vocabulary.index(word)] = 1.0
        row_tensor.append(word_encoding_0)
    label_value = label_encoding[row["class_name"]]
    return row_tensor, label_value


def get_vocabulary(text: pd.DataFrame, context_len):
    text["text"] = text["text"].str.lower()
    text_values = text["text"].values

    text_values_collection = [text.split(" ") for text in text_values]
    text_values_vocab = []
    for text in text_values_collection:
        text_values_vocab.extend(text[:context_len])
    text_values_vocab = list(set(text_values_vocab))
    return text_values_vocab
