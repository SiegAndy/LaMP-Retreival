from transformers import T5Tokenizer
from src.back_up.T5Dataset import T5Dataset
import pytorch_lightning as pl
import torch


class T5DataLoader(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        valid_data,
        tokenizer: T5Tokenizer,
        input_max_len: int = 1024,
        output_max_len: int = 10,
    ):
        super().__init__()
        self.valid_dataset = None
        self.train_dataset = None
        self.train_data = train_data
        self.valid_data = valid_data
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.out_max_len = output_max_len

    def setup(self, stage=None):
        self.train_dataset = T5Dataset(
            question=self.train_data["questions"],
            answer=self.train_data["labels"],
            tokenizer=self.tokenizer,
        )

        self.valid_dataset = T5Dataset(
            question=self.valid_data["questions"],
            answer=self.valid_data["labels"],
            tokenizer=self.tokenizer,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=2, num_workers=2, persistent_workers=True
        )
