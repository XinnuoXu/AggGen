#!/bin/bash

DSRC_PATH=../data_e2e/data-alg/src.dict
DTGT_PATH=../data_e2e/data-alg/tgt.dict
REL_PATH=../data_e2e/data-alg/relations.txt
DATA_PATH=/scratch/xxu/e2e/hmm_data/e2e
MODEL_PATH=/scratch/xxu/e2e/models/

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
	-visible_gpus 1 \
	-max_pos 512 \
	-alpha 0.9 \
	-state_emb_size 32 \
	-dec_hidden_size 128 \
	-dec_ff_size 256 \
	-enc_hidden_size 128 \
	-enc_ff_size 256 \
	-fake_global true \
	-log_file ../logs/val_abs_bert_cnndm \
	-result_path ../logs/abs_bert_cnndm
