dataset=$1
checkpoint=$2

python scripts/neg/finetune.py -bs 128 -d $dataset -t -ckpt $checkpoint -c scripts/neg/configs/t5_vae_32_max_none_none.yaml &&

python scripts/neg/finetune.py -bs 128 -d $dataset -t -ckpt $checkpoint  -c scripts/neg/configs/t5_vae_32_max_none_0.15.yaml &&

python scripts/neg/finetune.py -bs 128 -d $dataset -t -ckpt $checkpoint -c scripts/neg/configs/t5_vae_32_max_none_0.4.yaml &&

python scripts/neg/finetune.py -bs 128 -d $dataset -t -ckpt $checkpoint -c scripts/neg/configs/t5_vae_32_max_0.5_none.yaml &&

python scripts/neg/finetune.py -bs 128 -d $dataset -t -ckpt $checkpoint -c scripts/neg/configs/t5_vae_32_max_0.5_0.15.yaml &&

python scripts/neg/finetune.py -bs 128 -d $dataset -t -ckpt $checkpoint -c scripts/neg/configs/t5_vae_32_max_0.5_0.4.yaml &&

python scripts/neg/finetune.py -bs 128 -d $dataset -t -ckpt $checkpoint -c scripts/neg/configs/t5_vae_32_max_3_0.15.yaml &&

python scripts/neg/finetune.py -bs 128 -d $dataset -t -ckpt $checkpoint -c scripts/neg/configs/t5_vae_32_max_6_0.15.yaml
