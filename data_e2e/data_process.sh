DEFAULT_PYTHON_BIN=/home/xxu/.conda/envs/Transformer/bin/python

${DEFAULT_PYTHON_BIN} get_preprocess.py devel-e2e-tgt.txt
${DEFAULT_PYTHON_BIN} get_preprocess.py train-e2e-tgt.txt
${DEFAULT_PYTHON_BIN} get_preprocess.py test-e2e-tgt.txt

# Get srl parsing result
# wILL create 
	#`dev-webnlg-tgt.txt`
	#`train-webnlg-tgt.txt`
# in `data-srl/`
${DEFAULT_PYTHON_BIN} get_srl.py devel-e2e-tgt.txt
${DEFAULT_PYTHON_BIN} get_srl.py train-e2e-tgt.txt
${DEFAULT_PYTHON_BIN} get_srl.py test-e2e-tgt.txt

# Get trees
# Will create 
	#`webnlg_dev.jsonl`
	#`webnlg_train.jsonl` 
# in `data-tree/`
${DEFAULT_PYTHON_BIN} get_tree.py devel
${DEFAULT_PYTHON_BIN} get_tree.py train
${DEFAULT_PYTHON_BIN} get_tree.py test

# Get RST for training
# Will create
	#`webnlg_dev_src.jsonl`
	#`webnlg_dev_tgt.jsonl`
	#`webnlg_test_src.jsonl`
	#`webnlg_test_tgt.jsonl`
	#`webnlg_train_src.jsonl`
	#`webnlg_train_tgt.jsonl`
# in `data-rst/`
${DEFAULT_PYTHON_BIN} get_rst.py dev
${DEFAULT_PYTHON_BIN} get_rst.py train
${DEFAULT_PYTHON_BIN} get_rst.py test
${DEFAULT_PYTHON_BIN} get_postrst.py dev
${DEFAULT_PYTHON_BIN} get_postrst.py train
${DEFAULT_PYTHON_BIN} get_postrst.py test

# Get Alignment for RST
# Will create
	#`webnlg_dev_src.jsonl`
	#`webnlg_dev_tgt.jsonl`
	#`webnlg_test_src.jsonl`
	#`webnlg_test_tgt.jsonl`
	#`webnlg_train_src.jsonl`
	#`webnlg_train_tgt.jsonl`
# in `data-alg/`
${DEFAULT_PYTHON_BIN} get_alignment.py dev
${DEFAULT_PYTHON_BIN} get_alignment.py train
${DEFAULT_PYTHON_BIN} get_alignment.py test

# Get dictionaries for src relation
# in `data-alg/`
${DEFAULT_PYTHON_BIN} relation_dict.py

${DEFAULT_PYTHON_BIN} get_seq2seq.py devel
${DEFAULT_PYTHON_BIN} get_seq2seq.py train
${DEFAULT_PYTHON_BIN} get_seq2seq.py test
#${DEFAULT_PYTHON_BIN} get_dict_alg.py
${DEFAULT_PYTHON_BIN} get_dict.py

${DEFAULT_PYTHON_BIN} get_pretrain.py 
