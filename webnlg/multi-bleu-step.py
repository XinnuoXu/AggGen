#coding=utf8
import sys

black_list = ['[PAD]', '[CLS]']

if __name__ == '__main__':
    lower_bound = int(sys.argv[2])

    # SRC
    num_tripples = []
    for line in open('../logs/abs_bert_cnndm.' + sys.argv[1] + '.raw_src'):
        tokens = line.strip().split()
        tokens = [tok for tok in tokens if tok not in black_list][:-1]
        num_tripples.append(len(' '.join(tokens).split(' [SEP] ')))

    # CAND
    fpout = open('tmp/cands.txt', 'w')
    for i, line in enumerate(open('../logs/abs_bert_cnndm.' + sys.argv[1] + '.candidate')):
        if num_tripples[i] == lower_bound:
            fpout.write(line.strip()+'\n')
    fpout.close()

    # GOLD
    golds = [[], [], []]
    for i, line in enumerate(open('../logs/abs_bert_cnndm.' + sys.argv[1] + '.gold')):
        if num_tripples[i] == lower_bound:
            gs = line.strip().split("<ref_sep>")
            for j, g in enumerate(gs):
                golds[j].append(g.strip())

    for i in range(len(golds)):
        fpout = open('tmp/gold_'+str(i)+'.txt', 'w')
        for line in golds[i]:
            fpout.write(line+'\n')
        fpout.close()


