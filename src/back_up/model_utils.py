from transformers import T5Tokenizer
import torch
import numpy as np


def tokenize_dataset(tokenizer: T5Tokenizer, dataset: np.array, max_input_length):
    tokenized_datas = []
    for data in dataset:
        tokenized_data = tokenizer(data, max_length=max_input_length, truncation=True)
        tokenized_datas.append(tokenized_data)
        print(tokenized_data.shape)
        return torch.tensor(tokenized_datas)
