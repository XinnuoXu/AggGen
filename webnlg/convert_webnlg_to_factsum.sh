#!/bin/sh

#DEFAULT_PYTHON_BIN=/home/xxu/miniconda3/envs/Transformer/bin/python
#DEFAULT_PYTHON_BIN=~/opt/anaconda3/envs/factsum/bin/python
DEFAULT_PYTHON_BIN=${DEFAULT_PYTHON_PATH}

INPUT_FOLDER=$1
# Process train/dev splits
${DEFAULT_PYTHON_BIN} preprocess/converter.py --input ${INPUT_FOLDER}

# Process test seen and unseen splits
${DEFAULT_PYTHON_BIN} preprocess/converter.py --input ${INPUT_FOLDER} --partition test --categories seen
${DEFAULT_PYTHON_BIN} preprocess/converter.py --input ${INPUT_FOLDER} --partition test --categories unseen
