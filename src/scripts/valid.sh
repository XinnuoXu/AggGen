#!/bin/bash

DSRC_PATH=../webnlg/data-alg/src.dict
DTGT_PATH=../webnlg/data-alg/tgt.dict
REL_PATH=../webnlg/data-alg/relations.txt
DATA_PATH=/scratch/xxu/webnlg/tune_pretrain/hmm_data/webnlg
MODEL_PATH=/scratch/xxu/webnlg/tune_pretrain/models/

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
