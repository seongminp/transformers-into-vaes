import argparse

from base_models import datasets, models
from metrics import calc_au, calc_mi, calc_ppl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ckpt", "--checkpoint-path", help="Checkpoint path", default=None
    )
    parser.add_argument("-mt", "--metric", help="Metric to calculate")
    parser.add_argument("-c", "--config-file", help="Config file path")
    parser.add_argument("-d", "--dataset-name", help="Dataset to use")
    parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=1)
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
    # model.cuda()

    if args.metric == "mi":
        mi = calc_mi(model, dataloader, verbose=True)
        print("Mutual information:", mi)
    elif args.metric == "au":
        au = calc_au(model, dataloader, verbose=True)
        print("Active units:", au)
    elif args.metric == "ppl":
        ppl, nll, elbo, rec, kl = calc_ppl(model, dataloader, verbose=True)
        print(f"PPL: {ppl}, NLL: {nll}, -ELBO: {-elbo}, Rec: {rec}, KL: {kl}")
    else:
        print("Wrong metric")
