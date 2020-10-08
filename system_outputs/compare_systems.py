#coding=utf8

import pandas as pd
import re
import sys
import json
from os import path

PATTERN = r"(?P<relation>.+)\[(?P<object>.+)\]$"

def load_pairs(model_src_path, model_tgt_path, model_hmm_path):
    srcs = []
    for line in open(model_src_path):
        line = line.replace('[CLS]', '').replace('[PAD]', '').strip()
        src = ' [SEP] '.join(sorted([rec.strip().lower() for rec in line.split('[SEP]')][:-1]))
        srcs.append(src)
    cands = [line.strip() for line in open(model_tgt_path)]
    pairs = {}
    if not path.exists(model_hmm_path):
        for i in range(len(srcs)):
            pairs[srcs[i]] = (cands[i], '--')
    else:
        hmms = [line.strip() for line in open(model_hmm_path)]
        for i in range(len(srcs)):
            pairs[srcs[i]] = (cands[i], hmms[i])
    return pairs

def best_checkpoints():
    best_cp = {}
    best_cp['e2e.new.base'] = '22000'
    best_cp['e2e.new.hmm'] = '50000'
    best_cp['e2e.new.hmm_no_od'] = '50000'
    best_cp['e2e.new.hmm_no_ag'] = '50000'

    best_cp['webnlg.base'] = '17000'
    best_cp['webnlg.hmm'] = '30000'
    best_cp['webnlg.hmm_no_od'] = '30000'
    best_cp['webnlg.hmm_no_ag'] = '30000'

    best_cp['e2e.old.base'] = '50000'
    best_cp['e2e.old.hmm'] = '50000'
    best_cp['e2e.old.hmm_no_od'] = '50000'
    best_cp['e2e.old.hmm_no_ag'] = '50000'

    best_cp['e2e.cross.base'] = '50000'
    best_cp['e2e.cross.hmm'] = '50000'
    best_cp['e2e.cross.hmm_no_od'] = '50000'
    best_cp['e2e.cross.hmm_no_ag'] = '50000'

    return best_cp

if __name__ == "__main__":
    best_cps = best_checkpoints()
    SRC_PATH = './[TAG]/abs_bert_cnndm.[MODEL].raw_src'
    TGT_PATH = './[TAG]/abs_bert_cnndm.[MODEL].candidate'
    HMM_PATH = './[TAG]/abs_bert_cnndm.[MODEL].hmm'

    system_outputs = []
    for system in sys.argv[1:]:
        best_cp = best_cps[system]
        model_src_path = SRC_PATH.replace('[MODEL]', best_cp)
        model_tgt_path = TGT_PATH.replace('[MODEL]', best_cp)
        model_hmm_path = HMM_PATH.replace('[MODEL]', best_cp)

        model_src_path = model_src_path.replace('[TAG]', system)
        model_tgt_path = model_tgt_path.replace('[TAG]', system)
        model_hmm_path = model_hmm_path.replace('[TAG]', system)

        pairs = load_pairs(model_src_path, model_tgt_path, model_hmm_path)
        system_outputs.append(pairs)

    for src in system_outputs[0]:
        print ('<<SRC>>:', src)
        print ('====================================')
        for i in range(len(system_outputs)):
            if src not in system_outputs[i]:
                continue
            cand, hmm = system_outputs[i][src]
            print ('<<CAND>>['+sys.argv[i+1]+']:', cand)
            print ('<<HMM>>['+sys.argv[i+1]+']:', hmm)
            print ()
        print ()

