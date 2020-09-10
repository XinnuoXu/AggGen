#!/bin/bash

DSRC_PATH=../webnlg/data-alg/src.dict
DTGT_PATH=../webnlg/data-alg/tgt.dict
REL_PATH=../webnlg/data-alg/relations.txt
DATA_PATH=/scratch/xxu/webnlg/tune_pretrain/pretrain_data/webnlg
MODEL_PATH=/scratch/xxu/webnlg/tune_pretrain/pretrain_model/

python train.py \
	-mode vld_pretrain \
	-test_all \
	-data_path ${DATA_PATH} \
	-src_dict_path ${DSRC_PATH} \
        -tgt_dict_path ${DTGT_PATH} \
        -relation_path ${REL_PATH} \
	-model_path ${MODEL_PATH} \
	-batch_size 3000 \
	-test_batch_size 500 \
	-visible_gpus 1,2,3 \
	-max_pos 512 \
	-min_length 5 \
	-max_length 300 \
	-alpha 0.9 \
	-log_file ../logs/val_abs_bert_cnndm \
	-result_path ../logs/abs_bert_cnndm
