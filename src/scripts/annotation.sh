#!/bin/bash

DSRC_PATH=../data_webnlg/data-alg/src.dict
DTGT_PATH=../data_webnlg/data-alg/tgt.dict
REL_PATH=../data_webnlg/data-alg/relations.txt
DATA_PATH=/scratch/xxu/webnlg/tune_pretrain/ann_data/webnlg
MODEL_PATH=/scratch/xxu/webnlg/tune_pretrain/models/

python train.py \
	-mode annotation \
	-data_path ${DATA_PATH} \
	-src_dict_path ${DSRC_PATH} \
	-tgt_dict_path ${DTGT_PATH} \
	-relation_path ${REL_PATH} \
	-test_from ${MODEL_PATH}model_step_30000.pt \
	-test_batch_size 100 \
	-visible_gpus 0 \
	-max_pos 512 \
	-alpha 0.9 \
	-log_file ../logs/val_abs_bert_cnndm \
	-result_path ../logs/abs_bert_cnndm
