#!/bin/bash

RAW_PATH=../data_e2e.clean/data-pretrain/e2e_
REL_PATH=../data_e2e.clean/data-alg/relations.txt
JSON_PATH=/scratch/xxu/e2e.clean/pretrain_jsons/e2e

python preprocess.py \
	-mode pretrain_to_json \
	-raw_path ${RAW_PATH} \
	-relation_path ${REL_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 30 \
	-log_file ../logs/cnndm.log \
