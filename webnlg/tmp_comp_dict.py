#coding=utf8

import sys
import json

if __name__ == '__main__':
    with open('./data-seq/tgt.dict') as f:
        line = f.read().strip()
    d1 = json.loads(line)

    with open('./data-alg/tgt.dict') as f:
        line = f.read().strip()
    d2 = json.loads(line)

    missed_tok = set()
    for key in d1:
        if key not in d2:
            missed_tok.add(key)

    for line in open('data-seq/webnlg_train_tgt.jsonl'):
        line = line.strip().lower().replace('\t', ' ')
        flist = line.split()
        for item in flist:
            if item in missed_tok:
                print (line)
                break
