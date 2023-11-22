# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

# for training iemocap 4 way

python3 train.py --dataset="iemocap_4" --modalities="av" --epochs=55 --from_begin \
        --learning_rate=0.0001 --drop_rate=0.1 --seqcontext_nlayer=4 --gnn_nheads=7 \
        --log_in_tensorboard --ex_name="cogmen_av_iemocap_4_"