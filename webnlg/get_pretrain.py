#coding=utf8

def one_pair(src_file, tgt_file, src_out, tgt_out, tsk="train"):
    # Src
    src_tokens = []
    for line in open(src_file):
        flist = line.strip().split('\t')
        src_tokens.append('\t'.join(flist))
    # Tgt
    if tsk in ['train', 'dev']:
        tgt_tokens = ['\t'.join(line.strip().replace('-lrb-', '(').replace('-rrb-', ')').split('\t')[2:]) for line in open(tgt_file)]
    else:
        tgt_tokens = []
        for line in open(tgt_file):
            flist = line.strip().split('\t')
            cands = [item.split('[XXN]')[-1] for item in flist]
            tgt_tokens.append(' <REF_SEP> '.join(cands))
    # Writeout
    fpout = open(src_out, 'w')
    for line in src_tokens:
        fpout.write(line + '\n')
    fpout.close()

    fpout = open(tgt_out, 'w')
    for line in tgt_tokens:
        fpout.write(line + '\n')
    fpout.close()

if __name__ == '__main__':
    for tag in ['train', 'dev', 'test']:
        src_file = 'data-alg/webnlg_'+tag+'_src.jsonl'
        tgt_file = 'data-alg/webnlg_'+tag+'_tgt.jsonl'
        src_out = 'data-pretrain/webnlg_'+tag+'_src.jsonl'
        tgt_out = 'data-pretrain/webnlg_'+tag+'_tgt.jsonl'
        one_pair(src_file, tgt_file, src_out, tgt_out, tsk=tag)
