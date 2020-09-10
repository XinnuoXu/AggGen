#coding=utf8

import sys

def build_src_dict(input_file):
    tokens = {'[PAD]':0, '[SEP]':1, '[CLS]':2, '[UNK]':3, '[unused0]':4, '[unused1]':5, '[unused2]':6}
    for line in open(input_file):
        sentences = line.strip().split('\t')[:-1]
        for sent in sentences:
            for tok in sent.split(' '):
                tok = tok.lower()
                if tok not in tokens:
                    tokens[tok] = len(tokens)
    return tokens

def build_tgt_dict(input_file):
    tokens = {'[PAD]':0, '[SEP]':1, '[CLS]':2, '[UNK]':3, '[unused0]':4, '[unused1]':5, '[unused2]':6}
    for line in open(input_file):
        sentences = line.strip().split('\t')[2:]
        for sent in sentences:
            for tok in sent.split(' '):
                tok = tok.lower()
                if tok not in tokens:
                    tokens[tok] = len(tokens)
    return tokens

if __name__ == '__main__':
    import sys
    import os
    import json

    target_dir = './data-alg/'
    src_dict = build_src_dict(target_dir+'/webnlg_train_src.jsonl')
    tgt_dict = build_tgt_dict(target_dir+'/webnlg_train_tgt.jsonl')

    src_dict_fpout = open(target_dir+'/src.dict', 'w')
    tgt_dict_fpout = open(target_dir+'/tgt.dict', 'w')
    src_dict_fpout.write(json.dumps(src_dict))
    tgt_dict_fpout.write(json.dumps(tgt_dict))
    src_dict_fpout.close()
    tgt_dict_fpout.close()
