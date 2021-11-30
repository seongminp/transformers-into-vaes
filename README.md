# transformers-into-vaes

Code for [**_Finetuning Pretrained Transformers into Variational Autoencoders_**](https://aclanthology.org/2021.insights-1.5/) (our submission to NLP Insights Workshop 2021).

## Gathering data used in the paper:
1. Download all data (penn, snli, yahoo, yelp) from [this](https://github.com/ChunyuanLI/Optimus/blob/master/data/download_datasets.md) repository.

2. Change data path in `base_models.py` accordingly.

## Running experiments:

1. Install dependencies.
```bash
pip install -r requirements.txt
```

2. Run phase 1 (encoder only training):
```
./run_encoder_training snli
```

3. Run phase 2 (full training):
```bash
./run_training snli <path_to_checkpoint_from_phase_1>
```

## Calculating metrics:
```bash
python evaluate_all.py -d snli -bs 256 -c <path_to_config_file> -ckpt <path_to_checkpoint_file> 
```

