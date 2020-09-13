#coding=utf8

TAG='train'
#TAG='dev'

def load_tree():
    trees = {}
    for line in open('data-tree/webnlg_'+TAG+'.jsonl'):
        json_obj = json.loads(line.strip())
        summaries = json_obj["summary"]
        document = json_obj["document"]
        tokens = [item.split('|')[0] for item in document.split()]
        document = ' '.join(tokens)
        document = '\t'.join(document.split(' <TSP> '))
        trees[document] = '\t'.join(summaries)
    return trees

def read_raw_src():
    src_file = 'data/'+TAG+'-webnlg-src.txt'
    src_tokens = []
    for line in open(src_file):
        flist = line.strip().split(' ')
        tokens = [item.split('|')[0] for item in flist]
        string = ' '.join(tokens)
        src_tokens.append('\t'.join(string.split(' <TSP> ')))
    return src_tokens

def read_raw_tgt():
    tgt_file = 'data/'+TAG+'-webnlg-tgt.txt'
    tgt_tokens = [' '.join(line.strip().split(' ')[:-1]).replace(' <s> ', ' ') for line in open(tgt_file)]
    return tgt_tokens

def pair_data(srcs, tgts):
    pdata = {}
    for i in range(len(srcs)):
        pdata[srcs[i]] = tgts[i]
    return pdata

if __name__ == '__main__':
    import sys
    import json
    raw_srcs = read_raw_src()
    raw_tgts = read_raw_tgt()
    trees = load_tree()

    srcs = []
    for line in open('data-rst/webnlg_'+TAG+'_src.jsonl'):
        src = '\t'.join(line.strip().split('\t')[:-1])
        srcs.append(src)
    tgts = []
    for line in open('data-rst/webnlg_'+TAG+'_tgt.jsonl'):
        tgt = ' '.join(line.strip().split('\t')[1:]).replace('-lrb-', '(').replace('-rrb-', ')')
        tgts.append(tgt)

    pair_raw = pair_data(raw_srcs, raw_tgts)
    pair_alg = pair_data(srcs, tgts)

    mis_matching = 0
    for key in pair_raw:
        if pair_raw[key] != pair_alg[key]:
            #if abs(len(pair_raw[key].split()) - len(pair_alg[key].split())) > 2:
            print (pair_raw[key])
            print (pair_alg[key])
            print (trees[key])
            print ('\n\n')
            mis_matching += 1
    print ('mis_matching:', mis_matching)
