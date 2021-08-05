dataset=$1

# Max pooling.

## Dropout 0.15.
python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_max_none_0.15_yf.yaml -d $dataset -t -pe -eo &&
python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_max_none_0.15_nf.yaml -d $dataset -t -pe -eo &&

## No dropout.
python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_max_none_none_yf.yaml -d $dataset -t -pe -eo &&
python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_max_none_none_nf.yaml -d $dataset -t -pe -eo &&

## Dropout 0.4.
python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_max_none_0.4_yf.yaml -d $dataset -t -pe -eo &&
python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_max_none_0.4_nf.yaml -d $dataset -t -pe -eo &&

# Mean pooling.

## Dropout 0.15.
python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_mean_none_0.15_yf.yaml -d $dataset -t -pe -eo &&
python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_mean_none_0.15_nf.yaml -d $dataset -t -pe -eo

## No dropout.
#python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_mean_none_none_yf.yaml -d $dataset -t -pe -eo &&
#python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_32_mean_none_none_nf.yaml -d $dataset -t -pe -eo &&

## Different latents
#python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_64_max_none_0.15_yf.yaml -d $dataset -t -pe -eo &&
#python scripts/neg/finetune.py -bs 128 -c scripts/neg/pretraining_configs/t5_vae_128_max_none_0.15_yf.yaml -d $dataset -t -pe -eo
