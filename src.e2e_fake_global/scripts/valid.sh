#!/bin/bash

DSRC_PATH=../data_e2e/data-alg/src.dict
DTGT_PATH=../data_e2e/data-alg/tgt.dict
REL_PATH=../data_e2e/data-alg/relations.txt
DATA_PATH=/scratch/xxu/e2e/local_limit/hmm_data/e2e
MODEL_PATH=/scratch/xxu/e2e/local_limit/models.global/

python train.py \
	-mode validate \
	-test_all \
	-data_path ${DATA_PATH} \
	-src_dict_path ${DSRC_PATH} \
	-tgt_dict_path ${DTGT_PATH} \
	-relation_path ${REL_PATH} \
	-model_path ${MODEL_PATH} \
	-batch_size 140 \
	-test_batch_size 100 \
	-visible_gpus 0 \
	-max_pos 512 \
	-alpha 0.9 \
	-log_file ../logs/val_abs_bert_cnndm \
	-result_path ../logs/abs_bert_cnndm
