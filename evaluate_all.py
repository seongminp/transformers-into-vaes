import argparse
import re
import sys
from pathlib import Path

from base_models import datasets, models
from metrics import calc_all
from omegaconf import OmegaConf
from plotter import plot_line
from torch.utils.data import DataLoader
from tqdm import tqdm

checkpoint_pattern = re.compile(r"epoch=(\d+)-step=(\d+).ckpt")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cd", "--checkpoint-dir", help="Checkpoint directory", default=None
    )
    parser.add_argument(
        "-ckpt", "--checkpoint-path", help="Checkpoint path", default=None
    )
    parser.add_argument("-c", "--config-file", help="Config file path")
    parser.add_argument("-d", "--dataset-name", help="Dataset to use")
    parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("-p", "--plot", action="store_true", help="Plot stats")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config_file)

    base_model = models.get(conf.model_class)
    if not base_model:
        raise Exception("Wrong model.")

    model_class, tokenizer_class = (
        base_model["model_class"],
        base_model["tokenizer_class"],
    )
    tokenizer = tokenizer_class.from_pretrained(conf.base_model_name)

    dataset = datasets.get(args.dataset_name)
    if not dataset:
        raise Exception("Wrong dataset.")

    dataset_class = dataset["dataset_class"]
    out_dim = conf.out_dim
    test_set = (
        dataset_class(dataset["test_file"], tokenizer, out_dim)
        if dataset["test_file"]
        else None
    )
    dataloader = DataLoader(test_set, batch_size=args.batch_size)

    epochs, mis, aus = [], [], []

    def find_checkpoints(checkpoint_directory):

        ordered_checkpoints = {}

        for checkpoint_path in Path(checkpoint_directory).glob("**/*.ckpt"):
            match = checkpoint_pattern.match(checkpoint_path.name)
            if match:
                ordered_checkpoints[int(match.group(1))] = str(checkpoint_path)

        for key in sorted(ordered_checkpoints.keys()):
            yield key, ordered_checkpoints[key]

    if args.checkpoint_path:
        model = model_class.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            tokenizer=tokenizer,
            iterations_per_training_epoch=None,
            latent_dim=conf.latent_dim,
            pooling_strategy=conf.pooling_strategy,
            min_z=conf.min_z,
            fixed_reg_weight=None,
            base_model=conf.base_model_name,
        )
        model.eval()
        model.cuda()

        ppl, nll, elbo, rec, kl, mi, au = calc_all(model, dataloader, verbose=False)

        print(
            f"[{args.checkpoint_path}]"
            + f"PPL: {ppl}, NLL: {nll}, ELBO: {elbo}, REC: {rec}, KL: {kl},"
            + f"mi:{mi} au: {au}"
        )

        sys.exit()

    checkpoints = list(find_checkpoints(args.checkpoint_dir))

    for i, (key, checkpoint_path) in enumerate(tqdm(checkpoints)):
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            tokenizer=tokenizer,
            iterations_per_training_epoch=None,
            latent_dim=conf.latent_dim,
            pooling_strategy=conf.pooling_strategy,
            min_z=conf.min_z,
            fixed_reg_weight=None,
            base_model=conf.base_model_name,
        )
        model.eval()
        model.cuda()

        ppl, nll, elbo, rec, kl, mi, au = calc_all(model, dataloader, verbose=False)

        epochs.append(i + 1)
        mis.append(mi)
        aus.append(au)
        print(
            f"[{checkpoint_path}]"
            + f"PPL: {ppl}, NLL: {nll}, ELBO: {elbo}, REC: {rec}, KL: {kl},"
            + f"mi:{mi} au: {au}"
        )

    if args.plot:
        plot_line(
            "mi_and_au", "Epochs", "MI and AU", epochs, [("MI", mis), ("AU", aus)]
        )
