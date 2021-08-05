import argparse

from datamodule_finetune import LineByLineDataset
from datamodule_finetune_category import CategoryFinetuneDataset
from datamodule_pretrain import WikiDataset
from model_t5 import T5VAE
from transformers import T5TokenizerFast

# from datamodule_finetune import FinetuneDataset

models = {
    # "t5": {"model_class": T5ForConditionalGeneration, "tokenizer_class": T5TokenizerFast},
    "T5VAE": {"model_class": T5VAE, "tokenizer_class": T5TokenizerFast},
    # "t5iwae": {"model_class": T5IWAE, "tokenizer_class": T5TokenizerFast},
    # "t5denoise": {"model_class": T5Denoise, "tokenizer_class": T5TokenizerFast},
    # "t5fb": {"model_class": T5FB, "tokenizer_class": T5TokenizerFast},
    # "mass": {"model_class": MASS, "tokenizer_class": T5TokenizerFast},
    # "massvae": {"model_class": MASSVAE, "tokenizer_class": T5TokenizerFast},
    # "massiwae": {"model_class": MASSIWAE, "tokenizer_class": T5TokenizerFast},
    # "massdenoise": {"model_class": MASSDenoise, "tokenizer_class": T5TokenizerFast},
    # "massfb": {"model_class": MASSFB, "tokenizer_class": T5TokenizerFast},
}

datasets = {
    "wiki": {
        "dataset_class": WikiDataset,
        "train_file": "data/optimus/wikipedia.segmented.nltk.txt",
        "validate_file": None,
        "test_file": None,
        "train_dataset_size": 104213036,
    },
    "yelp": {
        "dataset_class": CategoryFinetuneDataset,
        "train_file": "data/optimus/yelp_data/train.txt",
        "validate_file": "data/optimus/yelp_data/valid.txt",
        "test_file": "data/optimus/yelp_data/test.txt",
        "train_dataset_size": 100000,
    },
    "snli": {
        "dataset_class": LineByLineDataset,
        "train_file": "data/optimus/snli_data/train.txt",
        "validate_file": "data/optimus/snli_data/valid.txt",
        "test_file": "data/optimus/snli_data/test.txt",
        "train_dataset_size": 100000,
    },
    "penn": {
        "dataset_class": LineByLineDataset,
        "train_file": "data/optimus/penn_data/train.txt",
        "validate_file": "data/optimus/penn_data/valid.txt",
        "test_file": "data/optimus/penn_data/test.txt",
        "train_dataset_size": 42068,
    },
    "yahoo": {
        "dataset_class": LineByLineDataset,
        "train_file": "data/optimus/yahoo_data/train.txt",
        "validate_file": "data/optimus/yahoo_data/valid.txt",
        "test_file": "data/optimus/yahoo_data/test.txt",
        "train_dataset_size": 100000,
    },
}
