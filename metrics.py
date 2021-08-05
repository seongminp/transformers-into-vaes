import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def calc_mi(model, test_data_batch, verbose=False):
    mi = 0
    num_examples = 0

    loop = tqdm(test_data_batch) if verbose else test_data_batch
    for batch in loop:
        batch_size = len(batch)
        num_examples += batch_size
        mutual_info = calc_batch_mi(model, batch)
        mi += mutual_info * batch_size

    return mi / num_examples


# From:
# https://github.com/jxhe/vae-lagging-encoder/blob/
# cdc4eb9d9599a026bf277db74efc2ba1ec203b15/modules/encoders/encoder.py
def calc_batch_mi(model, batch, verbose=False):
    """Approximate the mutual information between x and z
    I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
    Returns: Float
    """

    encoder_inputs, encoder_masks, labels = batch

    encoder_inputs = encoder_inputs.to(model.device)
    encoder_masks = encoder_masks.to(model.device)
    labels = labels.to(model.device)

    # [x_batch, nz]
    with torch.no_grad():
        _, z, mu, logvar = model(encoder_inputs, encoder_masks, labels)

    x_batch, nz = mu.size()

    # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
    neg_entropy = (
        -0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)
    ).mean()

    # [z_batch, 1, nz]
    z_samples = model.t5.reparameterize(mu, logvar)
    z_samples = z_samples.unsqueeze(1)

    # [1, x_batch, nz]
    mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
    var = logvar.exp()

    # (z_batch, x_batch, nz)
    dev = z_samples - mu

    # (z_batch, x_batch)
    log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (
        nz * math.log(2 * math.pi) + logvar.sum(-1)
    )

    # log q(z): aggregate posterior
    # [z_batch]
    log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

    return (neg_entropy - log_qz.mean(-1)).item()


def calc_au(model, test_dataloader, delta=0.01, verbose=False):
    """compute the number of active units"""

    data_loop = tqdm(test_dataloader) if verbose else test_dataloader

    def get_mu(batch):

        encoder_inputs, encoder_masks, labels = batch

        encoder_inputs = encoder_inputs.to(model.device)
        encoder_masks = encoder_masks.to(model.device)
        labels = labels.to(model.device)

        with torch.no_grad():
            _, _, mu, _ = model(encoder_inputs, encoder_masks, labels)

        return mu

    all_mu = [get_mu(batch) for batch in data_loop]

    mus = torch.vstack(all_mu)
    mu_mean = mus.mean(dim=0)

    vars = (mus - mu_mean).pow(2)
    au_var = vars.mean(dim=0)

    return (au_var >= delta).sum().item(), au_var


def calc_ppl(model, test_data_batch, verbose=False):
    total_recon_loss, total_reg_loss = 0, 0
    sentence_count, word_count = 0, 0

    loop = tqdm(test_data_batch) if verbose else test_data_batch

    for batch in loop:

        encoder_inputs, encoder_masks, labels = batch

        batch_size = encoder_inputs.shape[0]

        encoder_inputs = encoder_inputs.to(model.device)
        encoder_masks = encoder_masks.to(model.device)
        labels = labels.to(model.device)

        with torch.no_grad():
            logits, _, mu, logvar = model(encoder_inputs, encoder_masks, labels=labels)

        total_recon_loss += model.reconstruction_loss(logits, labels).sum()
        total_reg_loss += model.regularization_loss(mu, logvar).sum()

        sentence_count += batch_size
        word_count += encoder_masks.sum() - batch_size

    nll = (total_reg_loss + total_recon_loss) / sentence_count
    rec = total_recon_loss / sentence_count
    elbo = (total_reg_loss - total_recon_loss) / sentence_count
    kl = total_reg_loss / sentence_count
    ppl = torch.exp(nll * sentence_count / word_count)

    return ppl.item(), nll.item(), elbo.item(), rec.item(), kl.item()


def calc_all(model, test_dataloader, delta=0.01, verbose=False):

    loop = tqdm(test_dataloader) if verbose else test_dataloader

    total_mi = 0
    all_mu = []
    total_recon_loss, total_reg_loss = 0, 0
    sentence_count, word_count = 0, 0

    for batch in loop:

        encoder_inputs, encoder_masks, labels = batch

        batch_size = encoder_inputs.shape[0]

        encoder_inputs = encoder_inputs.to(model.device)
        encoder_masks = encoder_masks.to(model.device)
        labels = labels.to(model.device)

        with torch.no_grad():
            logits, z, mu, logvar = model(encoder_inputs, encoder_masks, labels=labels)

        # Mi
        latent_dim = mu.shape[0]
        neg_entropy = (
            -0.5 * latent_dim * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)
        ).mean()
        mu_expanded, logvar_expanded = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar_expanded.exp()
        dev = z.unsqueeze(1) - mu_expanded
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (
            latent_dim * math.log(2 * math.pi) + logvar_expanded.sum(-1)
        )
        log_qz = log_sum_exp(log_density, dim=1) - math.log(batch_size)
        mutual_info = (neg_entropy - log_qz.mean(-1)).item()
        total_mi += mutual_info * batch_size

        # AU
        all_mu.append(mu)

        # PPL, NLL, ELBO, REC, KL
        total_recon_loss += model.reconstruction_loss(logits, labels).sum()
        total_reg_loss += model.regularization_loss(mu, logvar).sum()
        sentence_count += batch_size
        word_count += encoder_masks.sum() - batch_size

    # MI
    mi = total_mi / sentence_count

    # AU
    mus = torch.vstack(all_mu)
    mu_mean = mus.mean(dim=0)
    vars = (mus - mu_mean).pow(2)
    au_var = vars.sum(dim=0) / (vars.shape[0] - 1)
    au = (au_var >= delta).sum().item()

    # PPL, NLL, ELBO, REC, KL
    nll = (total_reg_loss + total_recon_loss) / sentence_count
    rec = total_recon_loss / sentence_count
    elbo = (total_reg_loss - total_recon_loss) / sentence_count
    kl = total_reg_loss / sentence_count
    ppl = torch.exp(nll * sentence_count / word_count)

    return ppl.item(), nll.item(), elbo.item(), rec.item(), kl.item(), mi, au


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--checkpoint-path", help="Model checkpoint to use")
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
    train_set = dataset_class(dataset["train_file"], tokenizer, out_dim)
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
        base_model=conf.base_model_name,
        map_location="cpu",
    )
    model.eval()
    model.cuda()

    if args.metric == "mi":
        mi = calc_mi(model, dataloader)
        print("Mutual information:", mi)
    elif args.metric == "au":
        au = calc_au(model, dataloader)
        print("Active units:", au)
    elif args.metric == "ppl":
        ppl, nll, elbo, rec, kl = calc_ppl(model, dataloader)
        print(f"PPL: {ppl}, NLL: {nll}, -ELBO: {-elbo}, Rec: {rec}, KL: {kl}")
    else:
        print("Wrong metric")
