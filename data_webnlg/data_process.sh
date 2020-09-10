DEFAULT_PYTHON_BIN=/home/xx6/anaconda3/envs/Highlight/bin/python
# DEFAULT_PYTHON_BIN=/home/ik36/.conda/envs/factsum/bin/python
#DEFAULT_PYTHON_BIN=${DEFAULT_PYTHON_PATH}

#mv data/webnlg/dev-webnlg-src.txt data/
#mv data/webnlg/train-webnlg-src.txt data/
#mv data/webnlg/dev-webnlg-tgt.txt data/
#mv data/webnlg/train-webnlg-tgt.txt data/
#mv data/webnlg/test-seen-webnlg-src-unique.txt data/
#mv data/webnlg/test-unseen-webnlg-src-unique.txt data/
#mv data/webnlg/test-seen-reference0.lex data/
#mv data/webnlg/test-seen-reference1.lex data/
#mv data/webnlg/test-seen-reference2.lex data/
#mv data/webnlg/test-unseen-reference0.lex data/
#mv data/webnlg/test-unseen-reference1.lex data/
#mv data/webnlg/test-unseen-reference2.lex data/

# Preprocess
${DEFAULT_PYTHON_BIN} get_preprocess.py dev-webnlg-tgt.txt
${DEFAULT_PYTHON_BIN} get_preprocess.py train-webnlg-tgt.txt
${DEFAULT_PYTHON_BIN} get_preprocess.py test-seen-reference0.lex
${DEFAULT_PYTHON_BIN} get_preprocess.py test-seen-reference1.lex
${DEFAULT_PYTHON_BIN} get_preprocess.py test-seen-reference2.lex
${DEFAULT_PYTHON_BIN} get_preprocess.py test-unseen-reference0.lex
${DEFAULT_PYTHON_BIN} get_preprocess.py test-unseen-reference1.lex
${DEFAULT_PYTHON_BIN} get_preprocess.py test-unseen-reference2.lex

# Get srl parsing result
# wILL create 
	#`dev-webnlg-tgt.txt`
	#`train-webnlg-tgt.txt`
# in `data-srl/`
${DEFAULT_PYTHON_BIN} get_srl.py dev-webnlg-tgt.txt
${DEFAULT_PYTHON_BIN} get_srl.py train-webnlg-tgt.txt
${DEFAULT_PYTHON_BIN} get_srl.py test-seen-reference0.lex
${DEFAULT_PYTHON_BIN} get_srl.py test-seen-reference1.lex
${DEFAULT_PYTHON_BIN} get_srl.py test-seen-reference2.lex
${DEFAULT_PYTHON_BIN} get_srl.py test-unseen-reference0.lex
${DEFAULT_PYTHON_BIN} get_srl.py test-unseen-reference1.lex
${DEFAULT_PYTHON_BIN} get_srl.py test-unseen-reference2.lex

# Get trees
# Will create 
	#`webnlg_dev.jsonl`
	#`webnlg_train.jsonl` 
# in `data-tree/`
${DEFAULT_PYTHON_BIN} get_tree.py dev
${DEFAULT_PYTHON_BIN} get_tree.py train
${DEFAULT_PYTHON_BIN} get_tree.py test-seen 0
${DEFAULT_PYTHON_BIN} get_tree.py test-seen 1
${DEFAULT_PYTHON_BIN} get_tree.py test-seen 2
${DEFAULT_PYTHON_BIN} get_tree.py test-unseen 0
${DEFAULT_PYTHON_BIN} get_tree.py test-unseen 1
${DEFAULT_PYTHON_BIN} get_tree.py test-unseen 2

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
${DEFAULT_PYTHON_BIN} get_rst.py test-seen_0
${DEFAULT_PYTHON_BIN} get_rst.py test-seen_1
${DEFAULT_PYTHON_BIN} get_rst.py test-seen_2
${DEFAULT_PYTHON_BIN} get_rst.py test-unseen_0
${DEFAULT_PYTHON_BIN} get_rst.py test-unseen_1
${DEFAULT_PYTHON_BIN} get_rst.py test-unseen_2

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
${DEFAULT_PYTHON_BIN} get_alignment.py test-seen_0
${DEFAULT_PYTHON_BIN} get_alignment.py test-seen_1
${DEFAULT_PYTHON_BIN} get_alignment.py test-seen_2
${DEFAULT_PYTHON_BIN} get_alignment.py test-unseen_0
${DEFAULT_PYTHON_BIN} get_alignment.py test-unseen_1
${DEFAULT_PYTHON_BIN} get_alignment.py test-unseen_2

# Get dictionaries for src relation
# in `data-alg/`
${DEFAULT_PYTHON_BIN} relation_dict.py
${DEFAULT_PYTHON_BIN} get_dict_alg.py
${DEFAULT_PYTHON_BIN} get_dict.py

# Get test data from ref_0, ref_1, ref_2
# in `data-alg/`
${DEFAULT_PYTHON_BIN} merge_test.py

${DEFAULT_PYTHON_BIN} get_pretrain.py 
