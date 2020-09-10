#coding=utf8

TAG = 'train'
#TAG = 'dev'

def pair_data(srcs, tgts):
    pdata = {}
    for i in range(len(srcs)):
        pdata[srcs[i]] = tgts[i]
    return pdata

if __name__ == '__main__':
    import sys

    # pretrain_data
    srcs = []
    for line in open('data-pretrain/webnlg_'+TAG+'_src.jsonl'):
        src = '\t'.join(line.strip().split('\t')[:-1])
        srcs.append(src)
    tgts = []
    for line in open('data-pretrain/webnlg_'+TAG+'_tgt.jsonl'):
        tgt = ' '.join(line.strip().split('\t'))
        tgts.append(tgt)
    pair_alg = pair_data(srcs, tgts)

    # seq2seq data
    srcs = []
    for line in open('data-seq/webnlg_'+TAG+'_src.jsonl'):
        src = '\t'.join(line.strip().split('\t'))
        srcs.append(src)
    tgts = []
    for line in open('data-seq/webnlg_'+TAG+'_tgt.jsonl'):
        tgt = ' '.join(line.strip().split('\t'))
        tgts.append(tgt)
    pair_seq = pair_data(srcs, tgts)

    mis_matching = 0
    for key in pair_seq:
        if pair_seq[key] != pair_alg[key]:
            #if abs(len(pair_raw[key].split()) - len(pair_alg[key].split())) > 0:
            print (pair_seq[key])
            print (pair_alg[key] + '\n\n')
            mis_matching += 1
    print ('mis_matching:', mis_matching)
