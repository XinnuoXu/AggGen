#!/bin/bash

DSRC_PATH=../data_e2e.clean/data-alg/src.dict
DTGT_PATH=../data_e2e.clean/data-alg/tgt.dict
REL_PATH=../data_e2e.clean/data-alg/relations.txt
DATA_PATH=/scratch/xxu/e2e.clean/hmm_data/e2e
MODEL_PATH=/scratch/xxu/e2e.clean/models/
PRETRAIN_PATH=/scratch/xxu/e2e.clean/pretrain_model/

python train.py  \
	-mode train \
	-data_path ${DATA_PATH} \
	-src_dict_path ${DSRC_PATH} \
        -tgt_dict_path ${DTGT_PATH} \
	-relation_path ${REL_PATH} \
	-model_path ${MODEL_PATH} \
	-pretrain_path ${PRETRAIN_PATH}/model_step_23000.pt \
	-sep_optim true \
	-lr_tok 0.002 \
	-lr_hmm 0.01 \
	-train_steps 50000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps_tok 5000 \
	-warmup_steps_hmm 2000 \
	-accum_count 5 \
	-batch_size 140 \
	-report_every 50 \
	-state_emb_size 32 \
	-visible_gpus 0,1,2 \
	-log_file ../logs/abs_bert_cnndm
	#-share_emb \
	#-train_from ${MODEL_PATH}/model_step_38000.pt \
	#-fake_global true \
	#-dec_hidden_size 128 \
	#-dec_ff_size 256 \
	#-enc_hidden_size 128 \
	#-enc_ff_size 256 \
