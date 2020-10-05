#coding=utf8
import os, sys
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import kendalltau
from scipy.stats import spearmanr

CAND_HMM_PATH = '../logs/abs_bert_cnndm.30000.hmm'
SRC_PATH = '../Human_eva/scripts/webnlg_ann_src.jsonl'
GT_PATH = '../Human_eva/scripts/webnlg_ann_tgt.jsonl'

class NMI(object):

    def load_cand(self):
        cands = []
        for line in open(CAND_HMM_PATH):
            flist = line.strip().split('\t')
            types = flist[0].split()
            pattern = flist[1].split()
            group_id = 0
            clustering = {}
            for i in range(len(types)):
                clustering[types[i]] = group_id
                if pattern[i] == '1':
                    group_id += 1
            cands.append(clustering)
        return cands

    def load_gold(self):
        golds = []
        for line in open(GT_PATH):
            flist = line.strip().split('\t')
            one_example = []
            for item in flist:
                sequence = item.split('[XXN]')[1].split('|')
                clustering = {}
                for i, slot in enumerate(sequence):
                    types = slot.split('&')
                    for t in types:
                        clustering[t] = i
                one_example.append(clustering)
            golds.append(one_example)
        return golds

    def load_src(self):
        srcs = []
        for line in open(SRC_PATH):
            srcs.append(line.strip().split('\t')[-1].split('|'))
        return srcs

    def get_nmi(self, cands, golds, srcs):
        cand_scores = []
        ref_scores = []
        for i, src in enumerate(srcs):
            gold = golds[i]
            cand = cands[i]
            cand_sort = [cand[s] for s in src]

            gold_sort = []
            for g in gold:
                gs = [g[s] for s in src]
                gold_sort.append(gs)

            tmp_scores = [normalized_mutual_info_score(gold, cand_sort) for gold in gold_sort]
            cand_scores.append(max(tmp_scores))

            if len(gold_sort) == 2:
                ref_scores.append(normalized_mutual_info_score(gold_sort[0], gold_sort[1]))
            elif len(gold_sort) == 3:
                tmp_scores = []
                tmp_scores.append(normalized_mutual_info_score(gold_sort[0], gold_sort[1]))
                tmp_scores.append(normalized_mutual_info_score(gold_sort[0], gold_sort[2]))
                tmp_scores.append(normalized_mutual_info_score(gold_sort[1], gold_sort[2]))
                ref_scores.append(max(tmp_scores))
        return sum(cand_scores)/len(cand_scores), sum(ref_scores)/len(ref_scores)

    def run(self):
        cands = self.load_cand()
        golds = self.load_gold()
        srcs = self.load_src()
        cand_score, ref_score = self.get_nmi(cands, golds, srcs)
        print ('cand_score:', cand_score)
        print ('ref_score:', ref_score)


class Kendalltau(object):

    def load_cand(self):
        cands = []
        for line in open(CAND_HMM_PATH):
            flist = line.strip().split('\t')
            types = flist[0].split()
            pattern = flist[1].split()

            group_id = 0
            clustering = {}
            for i in range(len(types)):
                clustering[types[i]] = group_id
                if pattern[i] == '1':
                    group_id += 1
            cands.append(clustering)

            #clustering = {}
            #for i in range(len(types)):
            #    clustering[types[i]] = i
            #cands.append(clustering)
        return cands

    def load_gold(self):
        golds = []
        for line in open(GT_PATH):
            flist = line.strip().split('\t')
            one_example = []
            for item in flist:
                sequence = item.split('[XXN]')[1].split('|')
                clustering = {}
                for i, slot in enumerate(sequence):
                    types = slot.split('&')
                    for t in types:
                        clustering[t] = i
                one_example.append(clustering)
            golds.append(one_example)
        return golds

    def load_src(self):
        srcs = []
        for line in open(SRC_PATH):
            srcs.append(line.strip().split('\t')[-1].split('|'))
        return srcs

    def get_kendall(self, cands, golds, srcs):
        cand_scores = []
        ref_scores = []
        max_ref_scores = []
        for i, src in enumerate(srcs):
            gold = golds[i]
            cand = cands[i]
            cand_sort = [cand[s] for s in src]

            gold_sort = []
            for g in gold:
                gs = [g[s] for s in src]
                gold_sort.append(gs)

            if sum(cand_sort) == 0:
                continue
            tmp_scores = [kendalltau(gold, cand_sort)[0] for gold in gold_sort]
            max_score = max(tmp_scores)
            cand_scores.append(max(tmp_scores))

            if len(gold_sort) == 2:
                ref_scores.append(kendalltau(gold_sort[0], gold_sort[1])[0])
                max_ref_scores.append(kendalltau(gold_sort[0], gold_sort[1])[0])
            elif len(gold_sort) == 3:
                tmp_scores = []
                tmp_scores.append(kendalltau(gold_sort[0], gold_sort[1])[0])
                tmp_scores.append(kendalltau(gold_sort[0], gold_sort[2])[0])
                tmp_scores.append(kendalltau(gold_sort[1], gold_sort[2])[0])
                ref_scores.extend(tmp_scores)
                max_ref_scores.append(max(tmp_scores))
        return sum(cand_scores)/len(cand_scores), sum(ref_scores)/len(ref_scores), sum(max_ref_scores)/len(max_ref_scores)

    def run(self):
        cands = self.load_cand()
        golds = self.load_gold()
        srcs = self.load_src()
        cand_score, ref_score, max_ref_score = self.get_kendall(cands, golds, srcs)
        print ('cand_score:', cand_score)
        print ('ref_score:', ref_score)
        print ('max_ref_score:', max_ref_score)


if __name__ == '__main__':
    if sys.argv[1] == 'group':
        nmi_obj = NMI()
        nmi_obj.run()
    elif sys.argv[1] == 'sort':
        k_obj = Kendalltau()
        k_obj.run()
