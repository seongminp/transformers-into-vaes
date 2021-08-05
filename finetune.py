import argparse
import math
import os
import sys
from pathlib import Path

import pretty_errors
import pytorch_lightning as pl
import torch
import torch.nn as nn
from base_models import datasets, models
from generate import generate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="Config file path")
    parser.add_argument("-d", "--dataset-name", help="Dataset to use")
    parser.add_argument("-t", "--train", action="store_true", help="Run training")
    parser.add_argument(
        "-pe",
        "--pretrain-encoder",
        action="store_true",
        help="Run preliminary encoder training",
    )
    parser.add_argument(
        "-eo",
        "--encoder-only",
        action="store_true",
        help="Exit after encoder pretraining",
    )
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size", default=1)
    parser.add_argument(
        "-ckpt", "--checkpoint-path", help="Checkpoint path", default=None
    )
    args = parser.parse_args()

    experiment_name = (
        f"{args.dataset_name}_{str(Path(args.config_file).stem)}_bs{args.batch_size}"
    )
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
    validate_set = (
        dataset_class(dataset["validate_file"], tokenizer, out_dim)
        if dataset["validate_file"]
        else None
    )
    test_set = (
        dataset_class(dataset["test_file"], tokenizer, out_dim)
        if dataset["test_file"]
        else None
    )

    iterations_per_training_epoch = math.ceil(
        dataset["train_dataset_size"] / args.batch_size / torch.cuda.device_count()
    )

    model = model_class(
        tokenizer=tokenizer,
        iterations_per_training_epoch=iterations_per_training_epoch,
        latent_dim=conf.latent_dim,
        pooling_strategy=conf.pooling_strategy,
        min_z=conf.min_z,
        fixed_reg_weight=None,
        denoise_percentage=conf.denoise_percentage,
        base_model=conf.base_model_name,
    )

    cpu_count = os.cpu_count()
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, num_workers=cpu_count
    )
    val_dataloader = DataLoader(
        validate_set, batch_size=batch_size, num_workers=cpu_count
    )
    test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=cpu_count)

    if args.train:

        if args.checkpoint_path:

            model = model_class.load_from_checkpoint(
                args.checkpoint_path,
                strict=False,
                tokenizer=tokenizer,
                iterations_per_training_epoch=iterations_per_training_epoch,
                latent_dim=conf.latent_dim,
                pooling_strategy=conf.pooling_strategy,
                min_z=conf.min_z,
                fixed_reg_weight=None,
                denoise_percentage=conf.denoise_percentage,
                base_model=conf.base_model_name,
            )

            print(f"Loading checkpoint from: {args.checkpoint_path}")

        elif args.pretrain_encoder:

            experiment_suffix = "_enconly"

            model.fixed_reg_weight = 0
            if conf.freeze_decoder:
                model.freeze_decoder()
                # experiment_suffix += "_freezedec"

            early_stop_callback = EarlyStopping(
                monitor="val_recon_loss",
                min_delta=0.001,
                patience=15,
                verbose=True,
                mode="min",
                strict=True,
            )

            checkpoint_callback = ModelCheckpoint(
                monitor="val_recon_loss",
                mode="min",
                save_weights_only=True,
                save_top_k=15,
            )

            trainer = pl.Trainer(
                gpus=-1,
                accelerator="ddp",
                callbacks=[early_stop_callback, checkpoint_callback],
                max_epochs=15,
                plugins=DDPPlugin(
                    find_unused_parameters=True
                ),  # We ignore params from cross-attention.
                log_every_n_steps=1,
                logger=TensorBoardLogger(
                    save_dir=os.getcwd(),
                    version=experiment_name + experiment_suffix,
                    name="lightning_logs",
                ),
            )

            trainer.fit(
                model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
            )

            model = model_class.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                strict=False,
                tokenizer=tokenizer,
                iterations_per_training_epoch=iterations_per_training_epoch,
                latent_dim=conf.latent_dim,
                pooling_strategy=conf.pooling_strategy,
                min_z=conf.min_z,
                fixed_reg_weight=None,
                denoise_percentage=conf.denoise_percentage,
                base_model=conf.base_model_name,
            )

            print(
                "Finished preliminary encoder training.",
                f"Checkpoint saved at: {checkpoint_callback.best_model_path}",
            )

        if args.encoder_only:
            sys.exit(0)

        # Run regular training.
        early_stop_callback = EarlyStopping(
            # monitor="val_loss",
            monitor="finished_epoch",
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode="min",
            strict=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="finished_epoch",
            mode="max",
            save_weights_only=True,
            save_top_k=10,
        )

        trainer = pl.Trainer(
            gpus=-1,
            accelerator="ddp",
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=10,
            plugins=DDPPlugin(
                find_unused_parameters=True
            ),  # We ignore params from cross-attention.
            log_every_n_steps=1,
            logger=TensorBoardLogger(
                save_dir=os.getcwd(), version=experiment_name, name="lightning_logs"
            ),
        )
        trainer.fit(
            model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    elif args.checkpoint_path:
        model = model_class.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            map_location="cpu",
            tokenizer=tokenizer,
            iterations_per_training_epoch=None,
            latent_dim=conf.latent_dim,
            pooling_strategy=conf.pooling_strategy,
            fixed_reg_weight=None,
            denoise_percentage=conf.denoise_percentage,
            base_model=conf.base_model_name,
        )
        model.eval()
        model.to("cpu")

        fixed_strings = []

        # test_dataloader = DataLoader(test_set, batch_size=args.batch_size)
        test_dataloader = DataLoader(train_set, batch_size=args.batch_size)
        for tokenized, mask, label in test_dataloader:

            # category = category.to(model.master_ctx)
            # tokenized = tokenized.to(model.master_ctx)
            # mask = mask.to(model.master_ctx)

            # model.train()
            # x, z, mu, logvar = model(condition, tokenized, mask, label)
            # loss = x - 1
            # loss.mean().backward()
            # for name, param in model.named_parameters():
            #    if param.grad is None:
            #        print(name)

            # continue
            with torch.no_grad():

                fixed_tokens = generate(
                    model,
                    starter_tokens=[model.config.decoder_start_token_id],
                    input_ids=tokenized,
                    attention_mask=mask,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    top_p=0.95,
                    top_k=10,
                    temperature=1.0,
                    output_hidden_states=True,
                    num_beams=20,
                    use_cache=True,
                    # sampled_z = torch.ones((1, 64))
                )

            fixed = tokenizer.batch_decode(fixed_tokens, skip_special_tokens=True)
            original = tokenizer.batch_decode(tokenized, skip_special_tokens=True)

            for o, f in zip(original, fixed):

                # print(f"--------\n[CONDITION] {condition}\n[ORIGINAL] {o}\n[FIXED] {f}")
                print(f"--------\n[ORIGINAL] {o}\n[FIXED] {f}")
