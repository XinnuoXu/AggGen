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
    srcs = [line.strip() for line in open('data/test-seen-webnlg-src-unique.txt')]
    srcs = src_tokens(srcs)
    tgt0 = [line.strip() for line in open('data/test-seen-reference0.lex')]
    tgt1 = [line.strip() for line in open('data/test-seen-reference1.lex')]
    tgt2 = [line.strip() for line in open('data/test-seen-reference2.lex')]
    ori_data = {}
    for i, src in enumerate(srcs):
        t0 = tgt0[i].replace('\t', ' ')
        t1 = tgt1[i].replace('\t', ' ')
        t2 = tgt2[i].replace('\t', ' ')
        ori_data[src] = [t0, t1, t2]
    return ori_data

def read_tree(tag):
    input_srcs = [line.strip() for line in open('./data-alg/webnlg_'+tag+'_src.jsonl')]
    srcs = [' <TSP> '.join(src.split('\t')[:-1]) for src in input_srcs]
    tgts = [line.strip().split('\t') for line in open('./data-alg/webnlg_'+tag+'_tgt.jsonl')]
    tgts = [tgt[0]+'[XXN]'+tgt[1] if len(tgt) > 1 else tgt[0]+'[XXN]' for tgt in tgts]
    tree_data = {}
    for i, src in enumerate(srcs):
        tgt = tgts[i]
        input_src = input_srcs[i]
        tree_data[src] = [tgt, input_src]
    return tree_data

def merge_data(ori_data, tree_data):
    fpout_src = open('./data-alg/webnlg_test_src.jsonl', 'w')
    fpout_tgt = open('./data-alg/webnlg_test_tgt.jsonl', 'w')
    for src in ori_data:
        tree_0, alg_src = tree_data[0][src]
        tree_1, alg_src = tree_data[1][src]
        tree_2, alg_src = tree_data[2][src]
        tgt0, tgt1, tgt2 = ori_data[src]
        fpout_src.write(alg_src + '\n')
        tgt_str = [tree_0 + '[XXN]' + tgt0, tree_1 + '[XXN]' + tgt1, tree_2 + '[XXN]' + tgt2]
        fpout_tgt.write('\t'.join(tgt_str) + '\n')
    fpout_src.close()
    fpout_tgt.close()

if __name__ == '__main__':
    import sys
    test_refs = ['test-seen_0', 'test-seen_1', 'test-seen_2']
    ori_data = original_tgt()
    tree_data = [read_tree(tag) for tag in test_refs]
    merge_data(ori_data, tree_data)
