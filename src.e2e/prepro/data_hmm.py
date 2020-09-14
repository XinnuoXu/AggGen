import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import copy
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger
from others.tokenization import BertTokenizer

class HMMData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'

        with open(args.src_dict_path) as f:
            line = f.read().strip()
            self.src_dict = json.loads(line)

        with open(args.tgt_dict_path) as f:
            line = f.read().strip()
            self.tgt_dict = json.loads(line)

        with open(args.relation_path) as f:
            line = f.read().strip()
        self.relation_dict = json.loads(line)

        self.sep_vid = self.src_dict[self.sep_token]
        self.cls_vid = self.src_dict[self.cls_token]
        self.pad_vid = self.src_dict[self.pad_token]
        self.unk_vid = self.src_dict[self.unk_token]

    def convert_tokens_to_ids_tgt(self, tokens):
        return [self.tgt_dict[tok] if tok in self.tgt_dict else self.unk_vid for tok in tokens]

    def convert_tokens_to_ids_src(self, tokens):
        return [self.src_dict[tok] if tok in self.src_dict else self.unk_vid for tok in tokens]

    def preprocess_tgt(self, tgt, use_bert_basic_tokenizer=False):
        tgt = [' '.join(s).lower().split() for s in tgt]
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        tgt_len = sum([len(s) for s in tgt])
        if tgt_len < self.args.min_tgt_ntokens:
            return None
        if tgt_len > self.args.max_tgt_ntokens:
            return None
        if len(tgt) > self.args.max_tgt_fact:
            return None
        if self.args.tokenizer == 'sub-word':
            tgt_subtokens_strs = [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer))+ ' [unused1]' for tt in tgt]
            tgt_subtokens_strs.insert(0, '[unused0]')
        else:
            tgt_subtokens_strs = [' '.join(tt) + ' [unused1]' for tt in tgt]
            tgt_subtokens_strs.insert(0, '[unused0]')

        # Token to ID
        tgt_subtoken_idxs = [self.convert_tokens_to_ids_tgt(tgt_subtoken.split()) for tgt_subtoken in tgt_subtokens_strs]
        tgt_autogressive = []
        tgt_current = []
        for i in range(1, len(tgt_subtoken_idxs)):
            auto = []
            for j in range(0, i):
                auto.extend(tgt_subtoken_idxs[j])
            tgt_autogressive.append(auto)
            tgt_current.append(tgt_subtoken_idxs[i])
        return tgt_autogressive, tgt_current, tgt_txt


    def permutate_src(self, src_subtoken_list, relations, alignments, max_records_per_fact):

        def _permutate_uni(relations):
            uni_rel = [[rel] for rel in relations]
            return uni_rel

        def _permutate_bi(relations):
            bi_rel = []
            for i in range(len(relations)):
                for j in range(len(relations)):
                    if i == j:
                        continue
                    bi_rel.append([relations[i], relations[j]])
            return bi_rel
        
        def _permutate_tri(relations):
            tri_rel = []
            for i in range(len(relations)):
                for j in range(len(relations)):
                    for k in range(len(relations)):
                        if i == j or j == k or i == k:
                            continue
                        tri_rel.append([relations[i], relations[j], relations[k]])
            return tri_rel

        def _filter(must_in, must_out, cands):
            filted_cands = []
            for cand in cands:
                idx = 0
                while idx < len(must_in):
                    if must_in[idx] not in cand:
                        break
                    idx += 1
                if idx < len(must_in):
                    continue
                idx = 0
                while idx < len(must_out):
                    if must_out[idx] in cand:
                        break
                    idx += 1
                if idx < len(must_out):
                    continue
                filted_cands.append(cand)
            return filted_cands

        def _make_src(src_subtoken_list):
            new_subtoken_list = [[self.cls_token]]
            for src in src_subtoken_list:
                new_src = src + [self.sep_token]
                new_subtoken_list.append(new_src)
            src_subtokens = []
            for src in new_subtoken_list:
                src_subtokens.append(' '.join(src))
            src_subtokens = ' '.join(src_subtokens).split()
            # Token to ID
            src_subtoken_idxs = self.convert_tokens_to_ids_src(src_subtokens)
            return src_subtoken_idxs, new_subtoken_list

        def _mask_src(src_subtoken_list, relations, cand):
            p_mask = []
            p_mask.extend(len(src_subtoken_list[0]) * [False])
            for i in range(1, len(src_subtoken_list)):
                rel = relations[i-1]
                if rel in cand:
                    p_mask.extend(len(src_subtoken_list[i]) * [False])
                else:
                    p_mask.extend(len(src_subtoken_list[i]) * [True])
            return p_mask

        def _mask_src_each_step(src_subtoken_list, relations, step_cands):
            permutation_masks = []
            if len(step_cands) == 0:
                p_mask = _mask_src(src_subtoken_list, relations, [])
                permutation_masks.append(p_mask)
            else:
                for cand in step_cands:
                    p_mask = _mask_src(src_subtoken_list, relations, cand)
                    permutation_masks.append(p_mask)
            return permutation_masks

        if max_records_per_fact > 0:
            uni_rel = _permutate_uni(relations)
        else:
            uni_rel = []

        if max_records_per_fact > 1:
            bi_rel = _permutate_bi(relations)
        else:
            bi_rel = []

        if max_records_per_fact > 2:
            tri_rel = _permutate_tri(relations)
        else:
            tri_rel = []

        # Filter out the combined candidates by given alignments
        filted_cand_steps = []
        for i in range(len(alignments)):

            # Must contain
            if -1 in alignments[i]:
                must_in = []
            else:
                must_in = alignments[i]

            # Must not contain
            must_out = []
            for j in range(len(alignments)):
                if i == j:
                    continue
                must_out.extend(alignments[j])

            # Filter all candidates
            if len(must_in) >= max_records_per_fact:
                filted_cands = [must_in]
            else:
                if len(must_in) <= 1:
                    cands = uni_rel + bi_rel + tri_rel
                else:
                    cands = bi_rel + tri_rel
                filted_cands = _filter(must_in, must_out, cands)
            filted_cand_steps.append(filted_cands)

        src_subtoken_idxs, src_subtoken_list = _make_src(src_subtoken_list)

        # mask and combine srcs for each hidden state
        permutation_srcs = []
        for i, step_cands in enumerate(filted_cand_steps):
            permutation_src_mask = _mask_src_each_step(src_subtoken_list, relations, step_cands)
            permutation_srcs.append(permutation_src_mask)
            if len(step_cands) == 0:
                filted_cand_steps[i].append([self.relation_dict['lm_only']])
        return permutation_srcs, src_subtoken_idxs, filted_cand_steps


    def preprocess_src(self, src, relations, alignments, use_bert_basic_tokenizer=False, is_test=False, lower=True):
        if (lower):
            src = [' '.join(s).lower().split() for s in src]
        if ((not is_test) and len(src) == 0):
            return None
        if ((not is_test) and len(src) > self.args.max_src_nsents):
            return None
        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        if ((not is_test) and len(idxs) == 0):
            return None
        # Cut src
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None
        src_txt = [' '.join(sent) for sent in src]
        if self.args.tokenizer == 'sub-word':
            src_subtoken_list = [self.tokenizer.tokenize(text) for text in src_txt]
        else:
            src_subtoken_list = [text.split() for text in src_txt]

        if not is_test:
            if len(relations) <= len(alignments):
                max_records_per_fact = 1
            elif len(relations) == len(alignments) + 1:
                max_records_per_fact = 2
            else:
                max_records_per_fact = 3
        else:
            max_records_per_fact = 3
            alignments = [[-1]]

        permutation_srcs, src_subtoken_idxs, filted_cand_steps = self.permutate_src(src_subtoken_list, relations, alignments, max_records_per_fact)
        src_txt = [original_src_txt[i] for i in idxs]

        return permutation_srcs, src_subtoken_idxs, filted_cand_steps, src_txt


class PreproHMMData():
    '''
    @@ Called by preprocess.sh
    '''
    def __init__(self, args):
        self.args = args

    def _preprocess(self, params):
        corpus_type, json_file, args, save_file = params
        is_test = corpus_type == 'test'

        with open(args.relation_path) as f:
            line = f.read().strip()
        relation_dict = json.loads(line)

        if (os.path.exists(save_file)):
            logger.info('Ignore %s' % save_file)
            return
        bert = HMMData(args)
        logger.info('Processing %s' % json_file)
        jobs = json.load(open(json_file))
        lower = args.lower
        datasets = []
        for d in jobs:
            # Src_relations, tgt_alignments, tgt_trees
            src_r, tgt_a, tgt_t = d['src_r'], d['tgt_a'], d['tgt_t']
            relations = [relation_dict[r] for r in src_r]
            # DDDDDDDDDDDELETE
            #if len(relations) != 2:
            #    continue
            # DDDDDDDDDDDELETE_end
            if is_test:
                alignments = []
                for alg in tgt_a:
                    alignments.append(alg.split('|'))
            else:
                alignments = []
                for alg in tgt_a:
                    alg_list = alg.split('&&')
                    alignments.append([relation_dict[a] if a != '' else -1 for a in alg_list])

            # Process Src 
            source = d['src']
            b_data_src = bert.preprocess_src(source, relations, alignments,
                            use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, 
                            is_test=is_test, lower=lower)
            if b_data_src is None:
                continue
            permutation_src_masks, src_subtoken_idxs, comb_rels, src_txt = b_data_src

            if not is_test:
                # Process Tgt for train/dev
                tgt = d['tgt']
                tgt_length = sum([len(t) for t in tgt])
                b_data_tgt = bert.preprocess_tgt(tgt, 
                            use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                if b_data_tgt is None:
                    continue
                tgt_autogressive, tgt_current, tgt_txt = b_data_tgt
            else:
                # Process Tgt for test
                refs = d['tgt']
                tgt_autogressive = None
                tgt_current = None
                tgt_length = 0
                tgt_txt = [ref.lower() for ref in refs]

            # Read out
            b_data_dict = {"src": src_subtoken_idxs,
                           "src_mask": permutation_src_masks, 
                           "relations":relations,
                           "comb_rels": comb_rels,
                           "tgt": tgt_current, 
                           "tgt_atg": tgt_autogressive,
                           "tgt_len": tgt_length,
                           "src_txt": src_txt, "tgt_txt": tgt_txt}
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


class PreproHMMJson():
    '''
    @@ Called by preprocess_json.sh
    '''
    def __init__(self, args):
        self.args = args

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
            sorted_rel, sorted_rec = self._sort_src_sens(ex_src, relation, relation_dict)
            srcs.append(sorted_rec)
            relations.append(sorted_rel)
        return srcs, relations

    def _load_tgt(self, corpus_type):
        tgts = []; alignments = []; tree_struct = []
        root_tgt = self.args.raw_path + corpus_type + "_tgt.jsonl"
        for line in open(root_tgt):
            flist = line.strip().split('\t')
            fact_tree = flist[0]
            tree_struct.append(fact_tree)
            alignments.append(flist[1].split('|'))
            tgts.append([chunk.split() for chunk in flist[2:]])
        return tgts, alignments, tree_struct

    def _load_test(self, corpus_type):
        tgts = []; alignments = []; tree_struct = []
        root_tgt = self.args.raw_path + corpus_type + "_tgt.jsonl"
        for line in open(root_tgt):
            refs = line.strip().split('\t')
            refs_tree = []; refs_alg = []; refs_tgt = []
            for item in refs:
                ref = item.split('[XXN]')
                refs_tree.append(ref[0])
                refs_alg.append(ref[1])
                refs_tgt.append(ref[2])
            tree_struct.append(refs_tree)
            alignments.append(refs_alg)
            tgts.append(refs_tgt)
        return tgts, alignments, tree_struct 

    def preprocess(self):
        if (self.args.dataset != ''):
            datasets = [self.args.dataset]
        else:
            datasets = ['train', 'dev', 'test']

        with open(self.args.relation_path) as f:
            line = f.read().strip()
        relation_dict = json.loads(line)

        for corpus_type in datasets:
            srcs, relations = self._load_src(corpus_type, relation_dict) # Load src
            if corpus_type in ['train', 'dev']:
                tgts, alignments, tree_struct = self._load_tgt(corpus_type) # Load tgt
            else:
                tgts, alignments, tree_struct = self._load_test(corpus_type) # Load tgt for test

            json_objs = []
            for i, src in enumerate(srcs):
                tgt = tgts[i]
                src_r = relations[i]
                tgt_a = alignments[i]
                tgt_t = tree_struct[i]
                json_objs.append({'src': src, 'tgt': tgt, 'src_r': src_r, 'tgt_a': tgt_a, 'tgt_t': tgt_t})

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

