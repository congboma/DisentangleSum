#!/bin/bash

# Configure the resources required
#SBATCH -M volta                                                # use volta
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -A aiml
#SBATCH -n 1              	                                    # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 1              	                                    # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=24:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=16GB                                              # specify memory required per node (here set to 16 GB)

# Configure notifications
##SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
##SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
##SBATCH --mail-user=a1814291@adelaide.edu.au                    # Email to which notification will be sent

# record GPU utilisation
nvidia-smi -l > logs/nv-smi_sa.log.${SLURM_JOB_ID} 2>&1 &

# Execute your script (due to sequential nature, please select proper compiler as your script corresponds to)
time=$(date "+%Y%m%d-%H%M%S")
name=20230125_VTC_MultiNews_step1w_best_lr2_specFtWt0_ssFtWt.01_lossShSp.001

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
