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
name=20230126_val_1w_20230125_VTC_MultiNews_step1w_best_lr2_specFtWt0_ssFtWt.01_lossShSp.001

python translate.py -gpu 0 -batch_size 60 -beam_size 5 \
-model ./models/20230125_VTC_MultiNews_step1w_best_lr2_specFtWt0_ssFtWt.01_lossShSp.001/newser_step_10000.pt \
-src ../data/Multi_News_Gather/MultiNews500/valX.txt \
-output ./test-output/${name} -min_length 200 -max_length 300 \
-verbose -stepwise_penalty -coverage_penalty summary -beta 5 -length_penalty wu -alpha 0.9 \
-block_ngram_repeat 3 -replace_unk -report_rouge \
-ignore_when_blocking "." "</t>" "<t>" "story_separator_special_tag"
