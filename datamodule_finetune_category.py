import argparse

import torch
from torch.utils.data import Dataset


class CategoryFinetuneDataset(Dataset):
    def __init__(self, path, tokenizer, out_dim):

        self.tokenizer = tokenizer
        self.out_dim = out_dim
        with open(path, "r") as rf:
            self.data = rf.readlines()

    def __getitem__(self, index):
        return self.process_line(self.data[index])

    def __len__(self):
        return len(self.data)

    def process_line(self, line):

        line = line.strip()
        category, content = line.split("\t")

        encoded = self.tokenizer(
            # f"{category} {content}",
            f"{content}",
            add_special_tokens=True,
            max_length=self.out_dim,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        target_encoded = self.tokenizer(
            f"{content}",
            add_special_tokens=True,
            max_length=self.out_dim,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        encoder_input = encoded.input_ids.squeeze(0)
        encoder_mask = encoded.attention_mask.squeeze(0)
        decoder_target = target_encoded.input_ids.squeeze(0)

        return (
            encoder_input,
            encoder_mask,
            decoder_target,
        )


def get_category_embedding(category, out_dim):
    # Category embeddings is 12, because the hidden size of the decoder
    # has to be a multiple of number of attention heads (12).
    embedding = torch.zeros((out_dim, 12))
    if category == "positive":
        embedding[:, 0] = 1
        return embedding
    elif category == "negative":
        return embedding
    else:
        raise ValueError("Wrong category value:", category)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument.")
    args = parser.parse_args()
