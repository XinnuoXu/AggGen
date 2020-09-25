#!/bin/bash

DSRC_PATH=../data_e2e.clean/data-alg/src.dict
DTGT_PATH=../data_e2e.clean/data-alg/tgt.dict
REL_PATH=../data_e2e.clean/data-alg/relations.txt
DATA_PATH=/scratch/xxu/e2e.clean/pretrain_data/e2e
MODEL_PATH=/scratch/xxu/e2e.clean/pretrain_model/

python train.py \
	-mode test_pretrain \
	-data_path ${DATA_PATH} \
	-src_dict_path ${DSRC_PATH} \
	-tgt_dict_path ${DTGT_PATH} \
	-relation_path ${REL_PATH} \
	-test_from ${MODEL_PATH}model_step_49000.pt \
	-batch_size 3000 \
	-test_batch_size 500 \
	-visible_gpus 2 \
	-max_pos 512 \
	-min_length 7 \
	-max_length 200 \
	-alpha 0.9 \
	-log_file ../logs/val_abs_bert_cnndm \
	-result_path ../logs/abs_bert_cnndm
