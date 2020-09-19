#coding=utf8

'''
@@ This script ignores the tree structure
'''

if __name__ == '__main__':
    import sys
    input_dir = './data-rst/e2e_[DATA]_tgt.jsonl'
    output_dir = './data-rst/e2e_[DATA]_tgt_clean.jsonl'
    fpout = open(output_dir.replace('[DATA]', sys.argv[1]), 'w')
    for line in open(input_dir.replace('[DATA]', sys.argv[1])):
        line = line.strip()
        flist = line.strip().split('\t')
        new_flist = [flist[0]]
        accum_toks = []
        for i in range(1, len(flist)):
            toks = flist[i].split()
            if len(toks) < 3:
                accum_toks.extend(toks)
            else:
                new_flist.append(' '.join(accum_toks + toks))
                del accum_toks[:]
        if len(accum_toks) > 0:
            new_flist[-1] += ' ' + ' '.join(accum_toks)
        new_line = '\t'.join(new_flist)
        fpout.write(new_line + "\n")
    fpout.close()
