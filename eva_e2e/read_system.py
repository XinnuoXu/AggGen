#!/usr/bin/env python3
#coding=utf8

def read_candidates(path):
    lines = [line.strip() for line in open(path)]
    fpout = open('./tmp/cands.txt', 'w')
    for line in lines:
        fpout.write(line + '\n')
    fpout.close()

def read_gold(path):
    lines = [line.strip().split(' <ref_sep> ') for line in open(path)]
    fpout = open('./tmp/golds.txt', 'w')
    for gs in lines:
        for ref in gs:
            fpout.write(ref + '\n')
        fpout.write('\n')
    fpout.close()

if __name__ == '__main__':
    import sys
    cand_path = '../logs/abs_bert_cnndm.'+sys.argv[1]+'.candidate'
    gold_path = '../logs/abs_bert_cnndm.'+sys.argv[1]+'.gold'
    read_candidates(cand_path)
    read_gold(gold_path)
