#coding=utf8

import sys
sys.path.append('../src/')
from others.tokenization import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
use_bert_basic_tokenizer=False

def build_dict(input_file):
    tokens = {'[PAD]':0, '[SEP]':1, '[CLS]':2, '[UNK]':3, '[unused0]':4, '[unused1]':5, '[unused2]':6}
    for line in open(input_file):
        sentences = line.strip().split('\t')
        for sent in sentences:
            for tok in sent.split(' '):
                tok = tok.lower()
                if tok not in tokens:
                    tokens[tok] = len(tokens)
    return tokens

def build_dict_bert(input_file):
    tokens = {'[PAD]':0, '[SEP]':1, '[CLS]':2, '[UNK]':3, '[unused0]':4, '[unused1]':5, '[unused2]':6}
    for line in open(input_file):
        sentences = line.strip().split('\t')
        for sent in sentences:
            sent = sent.lower()
            sub_tokens = tokenizer.tokenize(sent, use_bert_basic_tokenizer=use_bert_basic_tokenizer)
            for tok in sub_tokens:
                if tok not in tokens:
                    tokens[tok] = len(tokens)
    return tokens

if __name__ == '__main__':
    import sys
    import os
    import json

    in_dir = './data-seq/'
    target_dir = './data-alg/'
    if sys.argv[1] == 'bert':
        src_dict = build_dict_bert(in_dir+'/e2e_train_src.jsonl')
        tgt_dict = build_dict_bert(in_dir+'/e2e_train_tgt.jsonl')
    else:
        src_dict = build_dict(in_dir+'/e2e_train_src.jsonl')
        tgt_dict = build_dict(in_dir+'/e2e_train_tgt.jsonl')

    src_dict_fpout = open(target_dir+'/src.dict', 'w')
    tgt_dict_fpout = open(target_dir+'/tgt.dict', 'w')
    src_dict_fpout.write(json.dumps(src_dict))
    tgt_dict_fpout.write(json.dumps(tgt_dict))
    src_dict_fpout.close()
    tgt_dict_fpout.close()

    merge_dict_fpout = open('./data-alg/merge.dict', 'w')
    for key in tgt_dict:
        if key in src_dict:
            continue
        src_dict[key] = len(src_dict)
    merge_dict_fpout.write(json.dumps(src_dict))
    merge_dict_fpout.close()
