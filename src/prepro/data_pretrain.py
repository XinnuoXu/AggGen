import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import copy
import torch
import subprocess
from collections import Counter
from os.path import join as pjoin
from multiprocess import Pool
from others.logging import logger

class PretrainData():
    def __init__(self, args):
        self.args = args

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        with open(args.src_dict_path) as f:
            line = f.read().strip()
            self.src_dict = json.loads(line)
        self.sep_vid = self.src_dict[self.sep_token]
        self.cls_vid = self.src_dict[self.cls_token]
        self.src_pad_vid = self.src_dict[self.pad_token]
        self.src_unk_vid = self.src_dict[self.unk_token]

        self.beg_token = '[unused0]'
        self.end_token = '[unused1]'
        with open(args.tgt_dict_path) as f:
            line = f.read().strip()
            self.tgt_dict = json.loads(line)
        self.beg_vid = self.tgt_dict[self.beg_token]
        self.end_vid = self.tgt_dict[self.end_token]
        self.tgt_pad_vid = self.tgt_dict[self.pad_token]
        self.tgt_unk_vid = self.tgt_dict[self.unk_token]

    def reshape_alignment(self, idxs, max_src_ntokens_per_sent, max_src_nsents, alignment):
        new_alignment = []
        for j in range(len(alignment)):
            alg = [alignment[j][i][:max_src_ntokens_per_sent] for i in idxs][:max_src_nsents]
            new_alignment.append(alg)
        return new_alignment

    def cls_alignment(self, alignment):
        new_alignment = []
        for j in range(len(alignment)):
            alg = []
            for item in alignment[j]:
                alg.append(sum(item)/len(item))
                alg.extend(item)
                alg.append(0.0)
            new_alignment.append(alg)
        return new_alignment

    def tokenize_alignment(self, src_subtokens, alignment):
        new_alignment = []
        for j in range(len(alignment)):
            alg = []; idx = 0
            for i, tok in enumerate(src_subtokens):
                if tok.startswith("##") and tok != "##":
                    alg.append(alg[-1])
                else:
                    alg.append(alignment[j][idx])
                    idx += 1
            new_alignment.append(alg)
        return new_alignment

    def tgt_alignment(self, alignment, tgt_subtokens_str):
        new_alignment = []; idx = 0
        alignment.append(copy.deepcopy(alignment[-1]))
        for i, tok in enumerate(tgt_subtokens_str.split(" ")):
            if tok.startswith("##") and tok != "##":
                new_alignment.append(copy.deepcopy(new_alignment[-1]))
            else:
                new_alignment.append(alignment[idx])
                idx += 1
        return new_alignment


    def preprocess(self, src, tgt, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        if len(idxs) == 0:
            return None

        # Src
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} '.format(self.sep_token).join(src_txt)
        src_subtokens = text.split(' ')
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        src_subtoken_idxs = [self.src_dict[tok] if tok in self.src_dict else self.src_unk_vid for tok in src_subtokens]
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        # Tgt
        #tgt_subtokens_str = '[unused0] ' + ' [unused1] '.join([' '.join(tt) for tt in tgt]) + ' [unused1]' + ' [unused2]'
        tgt_subtokens_str = '[unused0] ' + ' '.join([' '.join(tt) for tt in tgt]) + ' [unused1]'
        tgt_subtokens = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtokens) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = [self.tgt_dict[tok] if tok in self.tgt_dict else self.tgt_unk_vid for tok in tgt_subtokens]

        tgt_txt = '\t'.join([' '.join(tt) for tt in tgt])
        #src_txt = [original_src_txt[i] for i in idxs]
        src_txt = original_src_txt

        return src_subtoken_idxs, tgt_subtoken_idxs, segments_ids, src_txt, tgt_txt


class PreproPretrainData():
    def __init__(self, args):
        self.args = args

    def _preprocess(self, params):
        corpus_type, json_file, args, save_file = params
        is_test = corpus_type == 'test'
        if (os.path.exists(save_file)):
            logger.info('Ignore %s' % save_file)
            return

        bert = PretrainData(args)
        logger.info('Processing %s' % json_file)
        jobs = json.load(open(json_file))
        datasets = []
        for d in jobs:
            source, tgt = d['src'], d['tgt']
            # DDDDDDDDDDDELETE code
            #if is_test:
            #    if len(d['src']) < 5:
            #        continue
            # DDDDDDDDDDDELETE code end
            if (args.lower):
                source = [' '.join(s).lower().split() for s in source]
                tgt = [' '.join(s).lower().split() for s in tgt]
            b_data = bert.preprocess(source, tgt, is_test=is_test)
            if (b_data is None):
                continue
            src_subtoken_idxs, tgt_subtoken_idxs, segments_ids, src_txt, tgt_txt = b_data

            b_data_dict = {"src": src_subtoken_idxs,
                           "tgt": tgt_subtoken_idxs,
                           "segs": segments_ids,
                           "example_id": d['example_id'],
                           "src_txt": src_txt,
                           "tgt_txt": tgt_txt}
            datasets.append(b_data_dict)

        logger.info('Processed instances %d' % len(datasets))
        logger.info('Saving to %s' % save_file)
        torch.save(datasets, save_file)
        datasets = []
        gc.collect()

    def preprocess(self):
        if (self.args.dataset != ''):
            datasets = [self.args.dataset]
        else:
            datasets = ['dev', 'train', 'test']
        for corpus_type in datasets:
            a_lst = []
            for json_f in glob.glob(pjoin(self.args.raw_path, '*' + corpus_type + '.*.json')):
                real_name = json_f.split('/')[-1]
                a_lst.append((corpus_type, json_f, self.args, pjoin(self.args.save_path, real_name.replace('json', 'bert.pt'))))
            print(a_lst)
            pool = Pool(self.args.n_cpus)
            for d in pool.imap(self._preprocess, a_lst):
                pass
            pool.close()
            pool.join()


class PreproPretrainJson():
    def __init__(self, args):
        self.args = args

    def _sort_src_cross(self, ex_src, relation, relation_dict):
        tmp_rel = {}
        relation = relation.split('|')
        for rel in relation:
            tmp_rel[rel] = relation_dict[rel]
        rel_to_record = {}
        for i, record in enumerate(ex_src):
            if relation[i] in rel_to_record:
                rel_to_record[relation[i]].append(record)
            else:
                rel_to_record[relation[i]] = [record]
        sorted_rel = []
        sorted_rec = []
        for key, value in sorted(tmp_rel.items(), key = lambda d:d[1]):
            for i in range(len(rel_to_record[key])):
                sorted_rel.append(key)
            sorted_rec.extend(rel_to_record[key])
        return sorted_rel, sorted_rec

    def _sort_src_sens(self, ex_src, relation, relation_dict):
        tmp_rel = {}
        relation = relation.split('|')
        for rel in relation:
            tmp_rel[rel] = relation_dict[rel]
        rel_to_record = {}
        for i, record in enumerate(ex_src):
            rel_to_record[relation[i]] = record
        sorted_rel = []
        sorted_rec = []
        for key, value in sorted(tmp_rel.items(), key = lambda d:d[1]):
            sorted_rel.append(key)
            sorted_rec.append(rel_to_record[key])
        return sorted_rel, sorted_rec

    def _load_src(self, corpus_type, relation_dict):
        srcs = []; relations = []
        root_src = self.args.raw_path + corpus_type + "_src.jsonl"
        for line in open(root_src):
            flist = line.strip().split("\t")
            relation = flist[-1]
            ex_src = [sen.split() for sen in flist[:-1]]
            # sort by relation
            if self.args.cross_test:
                sorted_rel, sorted_rec = self._sort_src_cross(ex_src, relation, relation_dict)
            else:
                sorted_rel, sorted_rec = self._sort_src_sens(ex_src, relation, relation_dict)
            srcs.append(sorted_rec)
        return srcs

    def preprocess(self):
        if (self.args.dataset != ''):
            datasets = [self.args.dataset]
        else:
            datasets = ['train', 'test', 'dev']

        with open(self.args.relation_path) as f:
            line = f.read().strip()
        relation_dict = json.loads(line)

        for corpus_type in datasets:
            srcs = self._load_src(corpus_type, relation_dict)

            tgts = []
            root_tgt = self.args.raw_path + corpus_type + "_tgt.jsonl"
            for line in open(root_tgt):
                tgts.append([item.split() for item in line.strip().split('\t')])

            json_objs = []
            for i, src in enumerate(srcs):
                json_objs.append({'src': src, 'tgt': tgts[i], 'example_id':i})

            dataset = []
            p_ct = 0
            for d in json_objs:
                if (d is None):
                    continue
                dataset.append(d)
                if (len(dataset) > self.args.shard_size):
                    pt_file = "{:s}.{:s}.{:d}.json".format(self.args.save_path, corpus_type, p_ct)
                    with open(pt_file, 'w') as save:
                        save.write(json.dumps(dataset))
                        p_ct += 1
                        dataset = []
            if (len(dataset) > 0):
                pt_file = "{:s}.{:s}.{:d}.json".format(self.args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

