#!/bin/bash

python translate.py -gpu 0 -batch_size 60 -beam_size 5 \
-model ./models/20231215_VTC_MultiXSci_step1w_best_lr2_SF.01_SL.001_GPT2_min/newser_step_10000.pt \
-src ../data/Multi-XScience_utf8txt/testX.txt \
-output ./test-output/${name} -min_length 200 -max_length 300 \
-verbose -stepwise_penalty -coverage_penalty summary -beta 5 -length_penalty wu -alpha 0.9 \
-block_ngram_repeat 3 -replace_unk -report_rouge \
-ignore_when_blocking "." "</t>" "<t>" "story_separator_special_tag"
