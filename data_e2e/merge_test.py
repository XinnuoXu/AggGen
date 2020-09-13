#coding=utf8

def clean_record(record):
    toks = record.split()
    toks = [tok.split('|')[0] for tok in toks]
    return ' '.join(toks)

def src_tokens(srcs):
    ret = []
    for line in srcs:
        flist = line.split(' <TSP> ')
        src = []
        for rec in flist:
            rec = clean_record(rec)
            src.append(rec)
        ret.append(' <TSP> '.join(src))
    return ret

def original_tgt():
    srcs = [line.strip() for line in open('data/test-e2e-src.txt')]
    srcs = src_tokens(srcs)
    tgt = [line.strip() for line in open('data/test-e2e-tgt.txt')]
    ori_data = {}
    for i, src in enumerate(srcs):
        t = tgt[i].replace('\t', ' ').replace(' <s>', '')
        if src not in ori_data:
            ori_data[src] = []
        ori_data[src].append(t)
    return ori_data

def read_tree(tag):
    input_srcs = [line.strip() for line in open('./data-alg/e2e_'+tag+'_src.jsonl')]
    srcs = [' <TSP> '.join(src.split('\t')[:-1]) for src in input_srcs]
    tgts = [line.strip().split('\t') for line in open('./data-alg/e2e_'+tag+'_tgt.jsonl')]
    tgts = [tgt[0]+'[XXN]'+tgt[1] if len(tgt) > 1 else tgt[0]+'[XXN]' for tgt in tgts]
    tree_data = {}
    for i, src in enumerate(srcs):
        tgt = tgts[i]
        input_src = input_srcs[i]
        if src not in tree_data:
            tree_data[src] = []
        tree_data[src].append([tgt, input_src])
    return tree_data

def merge_data(ori_data, tree_data):
    fpout_src = open('./data-alg/e2e_test_src.jsonl', 'w')
    fpout_tgt = open('./data-alg/e2e_test_tgt.jsonl', 'w')
    rec_num = 0
    for src in ori_data:
        tgt_str = []
        for i in range(len(tree_data[src])):
            tree, alg_src = tree_data[src][i]
            tgt = ori_data[src][i]
            fpout_src.write(alg_src + '\n')
            tgt_str.append(tree + '[XXN]' + tgt)
            rec_num += 1
        fpout_tgt.write('\t'.join(tgt_str) + '\n')
    fpout_src.close()
    fpout_tgt.close()
    print ('Full record num: ', rec_num)

if __name__ == '__main__':
    import sys
    ori_data = original_tgt()
    tree_data = read_tree('test')
    merge_data(ori_data, tree_data)
