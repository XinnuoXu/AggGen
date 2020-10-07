#coding=utf8
import os, sys
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import kendalltau
from scipy.stats import spearmanr

GT_PATH = './annotation_human/rst_ann_gt_plan.jsonl'
SRC_PATH = './annotation_human/rst_ann_src.jsonl'

class Percision(object):
    def __init__(self, ANN_PATH):
        self.ANN_PATH = ANN_PATH

    def load_cand(self):
        cands = []
        for line in open(self.ANN_PATH):
            flist = line.strip().split('\t')
            sequence = flist[1].split('|')
            ann_res = {}
            for i, slot in enumerate(sequence):
                types = slot.split('&&')
                for t in types:
                    ann_res[t] = i
            cands.append(ann_res)
        return cands

    def load_cand_viterbi(self):
        cands = {}
        for line in open(self.ANN_PATH):
            flist = line.strip().split('\t')
            sequence = flist[1].split('|')
            example_id = int(flist[2])
            ann_res = {}
            for i, slot in enumerate(sequence):
                types = slot.split('&&')
                for t in types:
                    ann_res[t] = i
            cands[example_id] = ann_res
        return cands

    def load_gold(self):
        golds = []
        for line in open(GT_PATH):
            sequence = line.strip().split('|')
            ann_res = {}
            for i, slot in enumerate(sequence):
                types = slot.split('&&')
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
        precision = 0; num = 0
        for i, src in enumerate(srcs):
            gold = golds[i]
            if type(cands) is dict:
                if i not in cands:
                    print (i, src)
                    continue
            cand = cands[i]
            cand_ann = [cand[s] if s in cand else -1 for s in src]
            gold_ann = [gold[s] for s in src]
            for j in range(len(cand_ann)):
                if cand_ann[j] == gold_ann[j]:
                    precision += 1
                num += 1
        return precision/num

    def get_precision(self, cands, golds, srcs, ignore_miss=False):
        precision = 0; recall_num = 0; num = 0
        for i, src in enumerate(srcs):
            gold = golds[i]
            if type(cands) is dict:
                if i not in cands:
                    print (i, src)
                    continue
            cand = cands[i]
            cand_ann = [cand[s] if s in cand else -1 for s in src]
            gold_ann = [gold[s] for s in src]
            for j in range(len(cand_ann)):
                num += 1
                if cand_ann[j] == -1:
                    continue
                if cand_ann[j] == gold_ann[j]:
                    precision += 1
                recall_num += 1
        return precision/recall_num, recall_num/num

    def run(self, is_viterbi=False):
        if is_viterbi:
            cands = self.load_cand_viterbi()
        else:
            cands = self.load_cand()
        golds = self.load_gold()
        srcs = self.load_src()
        acc = self.get_acc(cands, golds, srcs)
        print ('ACC:', acc)
        precision, recall = self.get_precision(cands, golds, srcs)
        print ('Percision:', precision)
        print ('Recall:', recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        print ('F1:', f1)


if __name__ == '__main__':
    if sys.argv[1] == 'rule':
        ANN_PATH = './annotation_human/webnlg_ann_tgt.jsonl'
        precision_obj = Percision(ANN_PATH)
        precision_obj.run()
    elif sys.argv[1] == 'viterbi':
        ANN_PATH = '../logs/abs_bert_cnndm.30000.anno'
        precision_obj = Percision(ANN_PATH)
        precision_obj.run(is_viterbi=True)
    else:
        print ('ERROR: the parameter needs to be in [rule, viterbi]')
