#coding=utf8

import os
import sys
import json
import random

def one_candidate(src_path, tgt_path):
    srcs = []; tgts = []
    for line in open(src_path):
        src = line.strip().split('\t')
        tags = src[-1].split('|')
        src = src[:-1]
        src_obj = {}
        for i in range(len(tags)):
            src_obj[tags[i]] = src[i]
        srcs.append(src_obj)
    for line in open(tgt_path):
        tgt = line.strip().split('\t')
        if len(tgt) == 0:
            tgts.append(tgt)
        else:
            tgts.append(tgt[1:])
    return srcs, tgts

def phrase_filter(phrases):
    for i in range(1, len(phrases)):
        ph = phrases[i].split(' ')
        if len(ph) < 4:
            return False
    return True

def ex_filter(srcs0, tgts0, srcs1, tgts1, srcs2, tgts2):
    candidate_ids = []
    for i in range(len(srcs0)):
        if len(srcs0[i]) < 2:
            continue
        tgt0 = tgts0[i]
        tgt1 = tgts1[i]
        tgt2 = tgts2[i]
        if len(tgt0) == 0 or len(tgt1) == 0 or len(tgt2) == 0:
            continue
        patt0 = tgt0[0].split('|')
        patt1 = tgt1[0].split('|')
        patt2 = tgt2[0].split('|')
        if len(patt0) == 1 or len(patt1) == 1 or len(patt2) == 1:
            continue
        if phrase_filter(tgts0[i]) and phrase_filter(tgts1[i]) and phrase_filter(tgts2[i]):
            candidate_ids.append(i)
    return candidate_ids

def write_out(srcs0, tgts0, srcs1, tgts1, srcs2, tgts2, selected_ids):
    obj_list = []
    for idx in selected_ids:
        src = srcs0[idx]
        tgt_check0 = tgts0[idx][0].split('|')
        tgt0 = tgts0[idx][1:]
        tgt_check1 = tgts1[idx][0].split('|')
        tgt1 = tgts1[idx][1:]
        tgt_check2 = tgts2[idx][0].split('|')
        tgt2 = tgts2[idx][1:]
        obj = {}
        obj['SRC'] = src
        obj['TGT-0'] = tgt0
        obj['CHK-0'] = tgt_check0
        obj['TGT-1'] = tgt1
        obj['CHK-1'] = tgt_check1
        obj['TGT-2'] = tgt2
        obj['CHK-2'] = tgt_check2
        obj['ID'] = idx
        obj_list.append(obj)
    return obj_list

if __name__ == '__main__':
    BASE_PATH = '../../../data_webnlg/data-alg/'
    SAMPLE_NUM = 100

    src_path = BASE_PATH+'/webnlg_test-seen_0_src.jsonl'
    tgt_path = BASE_PATH+'/webnlg_test-seen_0_tgt.jsonl'
    srcs0, tgts0 = one_candidate(src_path, tgt_path)

    src_path = BASE_PATH+'/webnlg_test-seen_1_src.jsonl'
    tgt_path = BASE_PATH+'/webnlg_test-seen_1_tgt.jsonl'
    srcs1, tgts1 = one_candidate(src_path, tgt_path)

    src_path = BASE_PATH+'/webnlg_test-seen_2_src.jsonl'
    tgt_path = BASE_PATH+'/webnlg_test-seen_2_tgt.jsonl'
    srcs2, tgts2 = one_candidate(src_path, tgt_path)

    candidate_ids = ex_filter(srcs0, tgts0, srcs1, tgts1, srcs2, tgts2)
    selected_ids = random.sample(candidate_ids, SAMPLE_NUM)
    obj_list = write_out(srcs0, tgts0, srcs1, tgts1, srcs2, tgts2, selected_ids)
    
    fpout = open('../selected_example.json', 'w')
    fpout.write(json.dumps(obj_list))
    fpout.close()
