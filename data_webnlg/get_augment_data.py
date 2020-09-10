#coding=utf8

if __name__ == '__main__':
    import sys
    srcs = []
    record_num = []
    for line in open('./data-alg/webnlg_train_src.jsonl'):
        srcs.append(line.strip())
        num_realation = len(line.strip().split('\t')[-1].split('|'))
        record_num.append(num_realation)
    pairs = []
    for i, line in enumerate(open('./data-alg/webnlg_train_tgt.jsonl')):
        flist = line.strip().split('\t')
        num_phrase = len(flist[1].split('|'))
        if num_phrase == 1 and record_num[i] > 1 and record_num[i] < 4:
            pairs.append((srcs[i], line.strip()))
    fpout_src = open('./data-alg/webnlg_aug_src.jsonl', 'w')
    fpout_tgt = open('./data-alg/webnlg_aug_tgt.jsonl', 'w')
    for pair in pairs:
        fpout_src.write(pair[0]+'\n')
        fpout_tgt.write(pair[1]+'\n')
    fpout_src.close()
    fpout_tgt.close()
