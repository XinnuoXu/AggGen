#coding=utf8

import re
import csv
import stanza

BASE_PATH = './cleaned-data/'
OUT_PATH = './data/'

datasets = ['train', 'devel', 'test']
tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
PATTERN = r"(?P<relation>.+)\[(?P<object>.+)\]$"

def get_tgt(tgt):
    tgt_toks = tokenizer(tgt.strip()).sentences
    tgt = []
    for sentence in tgt_toks:
        tgt.append(' '.join([token.text for token in sentence.tokens]))
    tgt = ' <s> '.join(tgt) + ' <s>'
    return tgt

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

def preprocessing(dataset):
    input_path = BASE_PATH + '/' + dataset + '-fixed.csv'
    out_src_path = OUT_PATH + '/' + dataset + '-e2e-src.txt'
    out_tgt_path = OUT_PATH + '/' + dataset + '-e2e-tgt.txt'
    fpout_src = open(out_src_path, 'w')
    fpout_tgt = open(out_tgt_path, 'w')
    with open(input_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            src, subj = get_src(row[0])
            tgt = get_tgt(row[1])
            if subj == '' and dataset in ['train', 'devel']:
                continue
            fpout_src.write(src + '\n')
            fpout_tgt.write(tgt + '\n')
    fpout_src.close()
    fpout_tgt.close()

if __name__ == '__main__':
    for dataset in datasets:
        preprocessing(dataset)

