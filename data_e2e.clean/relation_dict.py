#coding=utf8

if __name__ == '__main__':
    import sys
    import json
    datasets = ['train', 'dev', 'test']
    relations = set(['hsmm_emission', 'lm_only', '<s>'])
    for ds in datasets:
        for line in open('data-alg/e2e_' + ds + '_src.jsonl'):
            relations |= set(line.strip().split('\t')[-1].split('|'))
    json_obj = {}
    for i, item in enumerate(relations):
        json_obj[item] = i
    fpout = open('data-alg/relations.txt', 'w')
    fpout.write(json.dumps(json_obj))
    fpout.close()
