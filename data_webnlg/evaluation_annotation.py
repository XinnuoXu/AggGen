#coding=utf8
import os, sys
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import kendalltau
from scipy.stats import spearmanr

GT_PATH = './data-rst/webnlg_ann_gt_plan.jsonl'
SRC_PATH = './data-rst/webnlg_ann_src.jsonl'
ANN_PATH = './data-alg/webnlg_ann_tgt.jsonl'

class Percision(object):

    def load_cand(self):
        cands = []
        for line in open(ANN_PATH):
            flist = line.strip().split('\t')
            sequence = flist[1].split('|')
            ann_res = {}
            for i, slot in enumerate(sequence):
                types = slot.split('&')
                for t in types:
                    ann_res[t] = i
            cands.append(ann_res)
        return cands

    def load_gold(self):
        golds = []
        for line in open(GT_PATH):
            sequence = line.strip().split('|')
            ann_res = {}
            for i, slot in enumerate(sequence):
                types = slot.split('&')
                for t in types:
                    ann_res[t] = i
            golds.append(ann_res)
        return golds

    def load_src(self):
        srcs = []
        for line in open(SRC_PATH):
            srcs.append(line.strip().split('\t')[-1].split('|'))
        return srcs

    def get_acc(self, cands, golds, srcs):
        acc = 0; num = 0
        for i, src in enumerate(srcs):
            gold = golds[i]
            cand = cands[i]
            cand_ann = [cand[s] for s in src]
            gold_ann = [gold[s] for s in src]
            for j in range(len(cand_ann)):
                if cand_ann[j] == gold_ann[j]:
                    acc += 1
                num += 1
        return acc/num

    def run(self):
        cands = self.load_cand()
        golds = self.load_gold()
        srcs = self.load_src()
        acc = self.get_acc(cands, golds, srcs)
        print ('acc:', acc)


if __name__ == '__main__':
    acc_obj = Percision()
    acc_obj.run()
