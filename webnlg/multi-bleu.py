#coding=utf8
import sys

if __name__ == '__main__':
    fpout = open('tmp/cands.txt', 'w')
    for line in open('../logs/abs_bert_cnndm.' + sys.argv[1] + '.candidate'):
        fpout.write(line.strip()+'\n')
    fpout.close()

    golds = [[], [], []]
    for line in open('../logs/abs_bert_cnndm.' + sys.argv[1] + '.gold'):
        gs = line.strip().split("<ref_sep>")
        for i, g in enumerate(gs):
            golds[i].append(g.strip())

    for i in range(len(golds)):
        fpout = open('tmp/gold_'+str(i)+'.txt', 'w')
        for line in golds[i]:
            fpout.write(line+'\n')
        fpout.close()


