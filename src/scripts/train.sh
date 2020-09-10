#!/bin/bash

DSRC_PATH=../data_webnlg/data-alg/src.dict
DTGT_PATH=../data_webnlg/data-alg/tgt.dict
REL_PATH=../data_webnlg/data-alg/relations.txt
DATA_PATH=/scratch/xxu/webnlg/tune_pretrain/hmm_data/webnlg
MODEL_PATH=/scratch/xxu/webnlg/tune_pretrain/models/
PRETRAIN_PATH=/scratch/xxu/webnlg/tune_pretrain/pretrain_model/

python train.py  \
	-mode train \
	-data_path ${DATA_PATH} \
	-src_dict_path ${DSRC_PATH} \
        -tgt_dict_path ${DTGT_PATH} \
	-relation_path ${REL_PATH} \
	-model_path ${MODEL_PATH} \
	-pretrain_path ${PRETRAIN_PATH}/model_step_17000.pt \
	-sep_optim true \
	-lr_tok 0.005 \
	-lr_hmm 0.01 \
	-accum_count 5 \
	-batch_size 140 \
	-report_every 50 \
	-train_steps 20000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps_tok 5000 \
	-warmup_steps_hmm 2000 \
	-visible_gpus 0,1,2 \
	-log_file ../logs/abs_bert_cnndm
	#-share_emb \
	#-train_from ${MODEL_PATH}/model_step_16000.pt \
