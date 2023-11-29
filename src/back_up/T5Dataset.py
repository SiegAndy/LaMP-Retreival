from transformers import T5Tokenizer
import torch


class T5Dataset:

    def __init__(self, question, answer, tokenizer: T5Tokenizer, input_max_len: int = 1024, output_max_len: int = 10):
        self.question = question
        self.answer = answer
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len

    def __len__(self):  # This method retrives the number of item from the dataset
        return len(self.question)

    def __getitem__(self, index):  # This method retrieves the item at the specified index item.
        cur_question = self.question[index]

        cur_answer = self.answer[index]

        input_tokenize = self.tokenizer(
            cur_question,
            add_special_tokens=True,
            max_length=self.input_max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        output_tokenize = self.tokenizer(
            cur_answer,
            add_special_tokens=True,
            max_length=self.output_max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"

        )

        input_ids = input_tokenize["input_ids"].flatten()
        attention_mask = input_tokenize["attention_mask"].flatten()
        labels = output_tokenize['input_ids'].flatten()

        out = {
            'question': cur_question,
            'answer': cur_answer,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': labels
        }

        return out
