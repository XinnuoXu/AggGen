#coding=utf8

TAG = 'train'
#TAG = 'dev'

def read_raw_src():
    src_file = 'data/'+TAG+'-e2e-src.txt'
    src_tokens = []
    for line in open(src_file):
        flist = line.strip().split(' ')
        tokens = [item.split('|')[0] for item in flist]
        string = ' '.join(tokens)
        src_tokens.append('\t'.join(string.split(' <TSP> ')))
    return src_tokens

def read_raw_tgt():
    tgt_file = 'data/'+TAG+'-e2e-tgt.txt'
    tgt_tokens = [' '.join(line.strip().split(' ')[:-1]).replace(' <s> ', ' ') for line in open(tgt_file)]
    return tgt_tokens

def pair_data(srcs, tgts):
    pdata = {}
    for i in range(len(srcs)):
        pdata[srcs[i]] = tgts[i]
    return pdata

if __name__ == '__main__':
    import sys
    raw_srcs = read_raw_src()
    raw_tgts = read_raw_tgt()
    srcs = []
    for line in open('data-pretrain/e2e_'+TAG+'_src.jsonl'):
        src = '\t'.join(line.strip().split('\t')[:-1])
        srcs.append(src)
    tgts = []
    for line in open('data-pretrain/e2e_'+TAG+'_tgt.jsonl'):
        #tgt = ' '.join(line.strip().split('\t')[2:]).replace('-lrb-', '(').replace('-rrb-', ')').replace('. .', '.')
        #tgt = ' '.join(line.strip().split('\t')).replace('-lrb-', '(').replace('-rrb-', ')')
        tgt = ' '.join(line.strip().split('\t'))
        tgts.append(tgt)

    pair_raw = pair_data(raw_srcs, raw_tgts)
    pair_alg = pair_data(srcs, tgts)

    mis_matching = 0
    for key in pair_raw:
        if pair_raw[key] != pair_alg[key]:
            #if abs(len(pair_raw[key].split()) - len(pair_alg[key].split())) > 0:
            print (pair_raw[key])
            print (pair_alg[key] + '\n\n')
            mis_matching += 1
    print ('mis_matching:', mis_matching)
