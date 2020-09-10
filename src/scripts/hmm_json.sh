#!/bin/bash

RAW_PATH=../data_webnlg/data-alg/webnlg_
REL_PATH=../data_webnlg/data-alg/relations.txt
JSON_PATH=/scratch/xxu/webnlg/tune_pretrain/hmm_jsons/webnlg

python preprocess.py \
	-mode hmm_to_json \
	-raw_path ${RAW_PATH} \
	-relation_path ${REL_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 30 \
	-log_file ../logs/cnndm.log \
