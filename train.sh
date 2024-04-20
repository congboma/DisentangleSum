#!/bin/bash

python train.py \
-save_model models/${name}/newser \
-data ../data/Multi-XScience_utf8txt/newser_pt/newser \
-word_vec_size 512 \
-rnn_size 512 -layers 4 -encoder_type transformer -decoder_type transformer -position_encoding \
-train_steps 10000 -warmup_steps 8000 -learning_rate 2 -decay_method noam -label_smoothing 0.1 \
-max_grad_norm 0 -dropout 0.2 \
-batch_size 4096 \
-optim adam -adam_beta2 0.998 -param_init 0 \
-batch_type tokens -normalization tokens -max_generator_batches 2 -accum_count 4 -share_embeddings \
-param_init_glorot -seed 777 -world_size 1 -gpu_ranks 0 -save_checkpoint_steps 10000 \
-loss_shared_wt 0.001 -loss_spec_wt 0.001 \
-copy_attn
