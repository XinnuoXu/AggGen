#coding=utf8

import pandas as pd
import re
import sys
import csv
import json

PATTERN = r"(?P<relation>.+)\[(?P<object>.+)\]$"

def process_src(doc):
    doc = doc.split(' <TSP> ')
    strings = []; relations = []
    for sentence in doc:
        sent_str = []
        relation = ""
        tripples = sentence.strip().split(' ')
        for item in tripples:
            tripple = item.split('|')
            tok = tripple[0]
            relation = tripple[2]
            sent_str.append(tok)
        strings.append(' '.join(sent_str))
        relations.append(relation)
    return strings, relations

def get_src(src):
    src = src.strip().split(', ')
    records = []
    subject = ""
    for s in src:
        m = re.match(PATTERN, s)
        r = m.group('relation').replace(' ', '')
        o = m.group('object')
        if r == 'name':
            subject = o
            continue
        sub = ['|'.join([item, 'ARG0', r]) for item in subject.split()]
        rel = ['|'.join([r, 'V', r])]
        obj = ['|'.join([item, 'ARG1', r]) for item in o.split()]
        record = ' '.join(sub + rel + obj)
        records.append(record)
    return ' <TSP> '.join(records), subject

def mapping_src(dataset_path, relation_dict):
    src_map = {}
    with open(dataset_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            org_src = row[0]
            src, subj = get_src(org_src)
            src, rels = process_src(src)
            p_src = ' [SEP] '.join(sorted([s.lower() for s in src]))
            src_map[p_src] = org_src
    return src_map

def create_csv(model_src_path, model_tgt_path, src_map):
    #raw_srcs = [line.replace('[CLS]', '').replace('[PAD]', '').strip() for line in open(model_src_path)]
    raw_srcs = []
    for line in open(model_src_path):
        line = line.replace('[CLS]', '').replace('[PAD]', '').strip()
        src = ' [SEP] '.join(sorted([rec.strip() for rec in line.split('[SEP]')][:-1]))
        raw_srcs.append(src)

    srcs = [src for src in raw_srcs]
    cands = [line.strip() for line in open(model_tgt_path)]
    new_srcs = []; new_cands = []
    for i in range(len(srcs)):
        src = srcs[i]
        cand = cands[i]
        if src not in src_map:
            print ("MISS:", src)
            continue
        new_srcs.append(src_map[src])
        new_cands.append(cand)
    d_data = {'mr':new_srcs, 'ref':new_cands}
    df = pd.DataFrame(d_data)
    df.to_csv('tmp.txt', index=False)

if __name__ == "__main__":
    dataset = sys.argv[1] # old/new
    if dataset == 'old':
        path = 'original-data/testset.csv'
        r_path = '../../data_e2e/data-alg/relations.txt'
    elif dataset == 'new':
        path = 'cleaned-data/test-fixed.csv'
        r_path = '../../data_e2e.clean/data-alg/relations.txt'

    model_src_path = '../../logs/[TAG]/abs_bert_cnndm.[MODEL].raw_src'.replace('[MODEL]', sys.argv[2])
    model_tgt_path = '../../logs/[TAG]/abs_bert_cnndm.[MODEL].candidate'.replace('[MODEL]', sys.argv[2])

    model_src_path = model_src_path.replace('[TAG]', sys.argv[3])
    model_tgt_path = model_tgt_path.replace('[TAG]', sys.argv[3])

    with open(r_path) as f:
        line = f.read().strip()
    relation_dict = json.loads(line)

    src_map = mapping_src(path, relation_dict)
    create_csv(model_src_path, model_tgt_path, src_map)

