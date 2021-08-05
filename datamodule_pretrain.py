import argparse

from torch.utils.data import IterableDataset


class WikiDataset(IterableDataset):
    def __init__(self, path, tokenizer, out_dim):

        self.tokenizer = tokenizer
        self.out_dim = out_dim
        self.file = open(path, "r")

    def __iter__(self):
        for line in self.file:

            encoded = self.tokenizer(
                line.strip(),
                add_special_tokens=True,
                return_tensors="pt",
            )

            if encoded.input_ids.shape[1] < 256:
                inp, mask, label = self.process_line(line)
                yield inp, mask, label

    def process_line(self, line):

        content = line.strip()
        # content = line.split("\t")

        encoded = self.tokenizer(
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

    def __exit__(self):
        self.file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("-bs", "--batch-size", type=int, help="Batch size", default=1)
    parser.add_argument("-l", "--out-dim", type=int, help="Output dimension", default=1)
    args = parser.parse_args()

    from multiprocessing import Pool

    from transformers import T5TokenizerFast

    data_file = "data/optimus/wikipedia.segmented.nltk.txt"

    class LineProcessor:
        def __init__(self, out_dim):
            self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
            self.out_dim = out_dim

        def __call__(self, line):
            encoded = self.tokenizer(
                line.strip(),
                add_special_tokens=True,
                return_tensors="pt",
            )
            return encoded.input_ids.shape[1] <= self.out_dim

    count = 0
    pool = Pool()
    with open(data_file, "r") as rf:
        for result in pool.map(LineProcessor(args.out_dim), rf):
            if result:
                count += 1

    print(f"Dataset length <= {args.out_dim}: {count+1}")
