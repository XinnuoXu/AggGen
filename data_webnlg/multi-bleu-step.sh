DEFAULT_PYTHON_BIN=/home/xxu/.conda/envs/Transformer/bin/python

echo 'Lower_bound=1'
${DEFAULT_PYTHON_BIN} multi-bleu-step.py $1 1
sh multi-bleu.sh

echo 'Lower_bound=2'
${DEFAULT_PYTHON_BIN} multi-bleu-step.py $1 2
sh multi-bleu.sh

echo 'Lower_bound=3'
${DEFAULT_PYTHON_BIN} multi-bleu-step.py $1 3
sh multi-bleu.sh

echo 'Lower_bound=4'
${DEFAULT_PYTHON_BIN} multi-bleu-step.py $1 4
sh multi-bleu.sh

echo 'Lower_bound=5'
${DEFAULT_PYTHON_BIN} multi-bleu-step.py $1 5
sh multi-bleu.sh

echo 'Lower_bound=6'
${DEFAULT_PYTHON_BIN} multi-bleu-step.py $1 6
sh multi-bleu.sh

echo 'Lower_bound=7'
${DEFAULT_PYTHON_BIN} multi-bleu-step.py $1 7
sh multi-bleu.sh

