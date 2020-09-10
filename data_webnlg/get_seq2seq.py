#coding=utf8

def one_pair(src_file, tgt_file, src_out, tgt_out):
    src_tokens = []
    for line in open(src_file):
        flist = line.strip().split(' ')
        tokens = [item.split('|')[0] for item in flist]
        string = ' '.join(tokens)
        src_tokens.append('\t'.join(string.split(' <TSP> ')))
    if type(tgt_file) == list:
        tgt_tokens_list = []
        for i, tfile in enumerate(tgt_file):
            #tgt_tokens = [' '.join(line.strip().split(' ')[:-1]).replace(' <s> ', '\t') for line in open(tfile)]
            #tgt_tokens_list.append(tgt_tokens)
            tgt_tokens_list.append([line.strip() for line in open(tfile)])
        tgt_merge = []
        #for item in tgt_tokens_list:
        #    print (len(item))
        for i in range(len(tgt_tokens_list[0])):
            tgt_merge.append(' <REF_SEP> '.join([llist[i] for llist in tgt_tokens_list]))
        tgt_tokens = tgt_merge
    else:
        tgt_tokens = [' '.join(line.strip().split(' ')[:-1]).replace(' <s> ', '\t') for line in open(tgt_file)]

    fpout = open(src_out, 'w')
    for line in src_tokens:
        fpout.write(line + '\n')
    fpout.close()

    fpout = open(tgt_out, 'w')
    for line in tgt_tokens:
        fpout.write(line + '\n')
    fpout.close()

if __name__ == '__main__':
    for tag in ['train', 'dev']:
        src_file = 'data/'+tag+'-webnlg-src.txt'
        tgt_file = 'data/'+tag+'-webnlg-tgt.txt'
        src_out = 'data-seq/webnlg_'+tag+'_src.jsonl'
        tgt_out = 'data-seq/webnlg_'+tag+'_tgt.jsonl'
        one_pair(src_file, tgt_file, src_out, tgt_out)
    src_file = 'data/test-seen-webnlg-src-unique.txt'
    tgt_file = ['data/test-seen-reference0.lex', 'data/test-seen-reference1.lex', 'data/test-seen-reference2.lex']
    src_out = 'data-seq/webnlg_test_src.jsonl'
    tgt_out = 'data-seq/webnlg_test_tgt.jsonl'
    one_pair(src_file, tgt_file, src_out, tgt_out)
