#coding=utf8

TAG = 'train'
#TAG = 'dev'

def process_summary(summary):
    toks = summary.strip().split()
    new_toks = []
    for item in toks:
        if item[0] == '(' or item[0] == ')':
            continue
        new_toks.append(item)
    summary = ' '.join(new_toks)
    return summary

def load_tree():
    trees = {}
    for line in open('data-tree/e2e_'+TAG+'.jsonl'):
        json_obj = json.loads(line.strip())
        summaries = json_obj["summary"]
        document = json_obj["document"]
        tokens = [item.split('|')[0] for item in document.split()]
        document = ' '.join(tokens)
        document = '\t'.join(document.split(' <TSP> '))
        summaries_str = ' '.join([process_summary(s) for s in summaries])
        trees[document] = (summaries, summaries_str)
    return trees

def pair_data(srcs, tgts):
    pdata = {}
    for i in range(len(srcs)):
        pdata[srcs[i]] = tgts[i]
    return pdata

if __name__ == '__main__':
    import sys
    import json
    srcs = []
    for line in open('data-pretrain/e2e_'+TAG+'_src.jsonl'):
        src = '\t'.join(line.strip().split('\t')[:-1])
        srcs.append(src)
    tgts = []
    for line in open('data-pretrain/e2e_'+TAG+'_tgt.jsonl'):
        tgt = ' '.join(line.strip().split('\t'))
        tgts.append(tgt)
    pair_alg = pair_data(srcs, tgts)

    trees = load_tree()

    mis_matching = 0
    for key in trees:
        if key not in pair_alg:
            continue
        if trees[key][1] != pair_alg[key]:
            #if abs(len(pair_raw[key].split()) - len(pair_alg[key].split())) > 0:
            print (trees[key][1])
            print (pair_alg[key] + '\n\n')
            mis_matching += 1
    print ('mis_matching:', mis_matching)
