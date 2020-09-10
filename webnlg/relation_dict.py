#coding=utf8

if __name__ == '__main__':
    import sys
    import json
    datasets = ['train', 'dev',
                'test-seen_0',
                'test-seen_1',
                'test-seen_2']
    relations = set(['hsmm_emission', 'lm_only', '<s>'])
    for ds in datasets:
        for line in open('data-alg/webnlg_' + ds + '_src.jsonl'):
            relations |= set(line.strip().split('\t')[-1].split('|'))
    json_obj = {}
    for i, item in enumerate(relations):
        json_obj[item] = i
    fpout = open('data-alg/relations.txt', 'w')
    fpout.write(json.dumps(json_obj))
    fpout.close()