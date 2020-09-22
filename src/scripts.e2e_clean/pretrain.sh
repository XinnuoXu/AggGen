#!/bin/bash

DSRC_PATH=../data_e2e/data-alg/src.dict
DTGT_PATH=../data_e2e/data-alg/tgt.dict
REL_PATH=../data_e2e/data-alg/relations.txt
DATA_PATH=/scratch/xxu/e2e/full/pretrain_data/e2e
MODEL_PATH=/scratch/xxu/e2e/full/pretrain_model/

python train.py  \
	-mode pretrain \
	-data_path ${DATA_PATH} \
	-src_dict_path ${DSRC_PATH} \
        -tgt_dict_path ${DTGT_PATH} \
	-relation_path ${REL_PATH} \
	-model_path ${MODEL_PATH} \
	-lr 0.1 \
	-save_checkpoint_steps 1000 \
	-batch_size 140 \
	-train_steps 50000 \
	-report_every 50 \
	-accum_count 5 \
	-seed 777 \
	-warmup_steps_tok 5000 \
	-warmup_steps_hmm 2000 \
	-max_pos 512 \
	-visible_gpus 0,1,2  \
	-log_file ../logs/abs_bert_cnndm
	#-share_emb \
