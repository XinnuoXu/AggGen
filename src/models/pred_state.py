#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import torch
from others.utils import tile


class State_predictor(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        self.s_beam_size = args.s_beam_size
        self.active_patt_num = args.active_patt_num
        self.extreme = args.extreme
        self.hsmm_sid = self.model.relation_dict['<s>']


    def _init_beam(self, init_logps):
        beam_size = self.s_beam_size
        active_size = (init_logps != -float("inf")).sum()
        beam_size = min(beam_size, active_size)

        topk_scores, topk_ids = init_logps.topk(beam_size, dim=-1)
        alive_seq = topk_ids.unsqueeze(1)
        topk_log_probs = topk_scores

        return alive_seq, topk_log_probs, beam_size


    def _one_beam_step(self, ex, score, trans_matrix, beam_size):
        masked_trans = trans_matrix.index_fill(1, ex, -float("inf"))
        last_s = ex[-1]
        curr_scores, curr_ids = masked_trans[last_s].topk(beam_size, dim=-1)
        cand_num = (curr_scores != -float("inf")).sum()
        ex = tile(ex.unsqueeze(0), cand_num, dim=0)
        curr_ids = curr_ids[:cand_num].unsqueeze(1)
        seqs = torch.cat((ex, curr_ids), dim=1)
        probs = curr_scores[:cand_num] + score
        return seqs, probs


    def _get_best_state_sequence(self, relation):

        # Pruing search for best state sequences
        init_matrix, trans_matrix = self.model.trans_logprobs()

        # Mask init matrix
        init_mask = torch.tensor(self.model.relation_size * [-float("inf")], device=self.device)
        index = torch.tensor(relation, device=self.device)
        init_mask.index_fill_(0, index, 0.0)
        init_logps = init_matrix + init_mask

        # Mask transition matrix
        self_mask = torch.tensor(self.model.relation_size * [-float("inf")], device=self.device)
        self_mask = torch.diag(self_mask)
        in_mask = torch.Tensor(self.model.relation_size, self.model.relation_size).fill_(-float("inf")).to(self.device)
        in_mask.index_fill_(1, index, 0.0)
        torch.set_printoptions(profile="full")
        trans_logps = trans_matrix + self_mask + in_mask

        # Initialize the beam
        alive_seq, topk_log_probs, beam_size = self._init_beam(init_logps)
        hypotheses = []; scores = []

        # Beam search
        while alive_seq.size(1) < len(relation):
            new_alive_seq = []
            new_topk_log_probs = []
            for i in range(beam_size):
                seqs, probs = self._one_beam_step(alive_seq[i], topk_log_probs[i], trans_logps, beam_size)
                new_alive_seq.append(seqs)
                new_topk_log_probs.append(probs)
            alive_seq = torch.cat(new_alive_seq, dim=0)
            topk_log_probs = torch.cat(new_topk_log_probs, dim=0)
            topk_log_probs, select_idx = topk_log_probs.topk(beam_size, dim=-1)
            alive_seq = alive_seq.index_select(0, select_idx)

        # Get the best sequence
        for i in range(min(alive_seq.size(0), self.active_patt_num)):
            hypotheses.append(alive_seq[i])
            scores.append(topk_log_probs[i])

        return hypotheses, scores


    def _binary_pattern(self, max_f, extreme):
        def _filt_cands(cand):
            seq_zeros = 0
            for i in range(len(cand)):
                if cand[i] == 0:
                    seq_zeros += 1
                elif cand[i] == 1:
                    if seq_zeros > 2:
                        return False
                    seq_zeros = 0
            return True

        if extreme == 1:
            cands = [[1]*max_f]
        else:
            k = max_f-1
            bin_k = lambda x : [(x >> i) & 1 for i in range(k)]
            cands = []
            for i in range(1<<k):
                l = bin_k(i)
                l.reverse()
                cand = l+[1]
                if _filt_cands(cand):
                    cands.append(cand)
        if extreme == -1:
            filted_cands = []
            for item in cands:
                if max_f > 3:
                    if sum(item) == max_f-1:
                        filted_cands.append(item)
                else:
                    if sum(item) == max_f:
                        filted_cands.append(item)
            cands = filted_cands
        elif extreme == -2:
            filted_cands = []
            for item in cands:
                if max_f > 4:
                    if sum(item) == max_f-2:
                        filted_cands.append(item)
                elif max_f > 3:
                    if sum(item) == max_f-1:
                        filted_cands.append(item)
                else:
                    if sum(item) == max_f:
                        filted_cands.append(item)
            cands = filted_cands
        return cands


    def _get_src_masks(self, seq, bpatts, cand_states, masks, min_f, max_f, patt_score):
        src_masks = []; seqs = []; bi_patts = []; patt_scores = []
        for k, bp in enumerate(bpatts):
            emi_num = sum(bp)
            if emi_num < min_f or emi_num > max_f:
                continue
            cstate = []; s_mask = []
            for i in range(len(bp)):
                cstate.append(int(seq[i]))
                if bp[i] == 1:
                    try:
                        idx = cand_states.index(cstate)
                    except:
                        del s_mask[:]
                        break
                    s_mask.append(masks[idx])
                    del cstate[:]
            if len(s_mask) > 0:
                src_masks.append(s_mask)
                seqs.append(seq)
                bi_patts.append(bp)
                patt_scores.append(patt_score[k])
        return src_masks, seqs, bi_patts, patt_scores


    def _binary_patt_ranker(self, seq, bpatts, min_f, max_f):
        ext_logps = self.model.external_logprobs()

        # get combined stated transition pairs
        step_states = []; new_bpatts = []
        for bp in bpatts:
            emi_num = sum(bp)
            if emi_num < min_f or emi_num > max_f:
                continue
            new_bpatts.append(bp)
            states = []; st = []
            for i in range(len(bp)):
                st.append(int(seq[i]))
                if bp[i] == 1:
                    states.append(st)
                    st = []
            step_states.append(states)

        # get score for each binary pattern
        bi_patt_scores = []
        for bp in step_states:
            score = 0
            for t in range(len(bp)):
                if t == 0:
                    pre_s = self.hsmm_sid
                else:
                    pre_s = bp[t-1][-1]
                s = bp[t][0]
                score += ext_logps[pre_s][s]
            bi_patt_scores.append(score/float(len(bp)))
        bi_patt_scores = torch.stack(bi_patt_scores)

        # get topk pattern
        k = min(self.args.agg_topk, bi_patt_scores.size(0))
        topk_scores, topk_ids = bi_patt_scores.topk(k, dim=-1)
        topk_patts = [new_bpatts[idx] for idx in topk_ids]

        return topk_patts, topk_scores


    def _one_example(self, r_num, masks, cand_states, relation):
        # Get best state sequences
        state_sequences, state_scores = self._get_best_state_sequence(relation)
        # Get binary patterns
        max_f = r_num
        min_f = int((r_num - 0.5)/3)+1
        bpatts = self._binary_pattern(max_f, self.extreme)
        # Get src masks for token decoding
        src_masks = []; state_seqs = []; binary_patts = []; patt_scores = []
        for i, seq in enumerate(state_sequences):
            seq_score = state_scores[i]
            bps, bps_score = self._binary_patt_ranker(seq, bpatts, min_f, max_f)
            patt_score = seq_score+bps_score
            res = self._get_src_masks(seq, bps, cand_states, masks, min_f, max_f, patt_score)
            s_masks, s_seqs, b_patts, p_scores = res
            src_masks.extend(s_masks)
            state_seqs.extend(s_seqs)
            binary_patts.extend(b_patts)
            patt_scores.extend(p_scores)
        return src_masks, state_seqs, binary_patts, patt_scores


    def state_predictor(self, batch):
        pmt_msk = batch.pmt_msk
        states = batch.states
        ex_idx = batch.ex_idx
        relations = batch.relations
        #record_num = [len(ex_src) for ex_src in batch.src_str]
        record_num = [len(rel) for rel in relations]
        bsz = len(ex_idx)
        batch_states = []
        for i, example in enumerate(record_num):
            r_num = record_num[i]
            s_idxs = ex_idx[i][0]
            relation = relations[i]
            masks = pmt_msk[s_idxs[0]:s_idxs[1]]
            cand_states = states[s_idxs[0]:s_idxs[1]]
            res = self._one_example(r_num, masks, cand_states, relation)
            #src_masks, state_seqs, binary_patts, patt_scores = res
            batch_states.append(res)

        return batch_states


