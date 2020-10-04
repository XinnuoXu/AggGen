#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import re
import math
import torch

from tensorboardX import SummaryWriter
from others.utils import rouge_results_to_str, test_rouge, tile


def build_annotator(args, model, logger=None):
    annotator = Annotator(args, model, logger=logger)
    return annotator

class Vierbi(object):
    def __init__(self, model, device):
        self.device = device
        self.model = model
        relation_dict = self.model.relation_dict
        self.hsmm_sid = relation_dict['<s>']

    def recover_bps(self, gama, arg_gama, step_states):
        idx = gama[-1].index(max(gama[-1]))
        viterbi_seq = [step_states[-1][idx]]
        for i in range(len(arg_gama)-1, 1, -1):
            idx = arg_gama[i][idx]
            viterbi_seq.insert(0, step_states[i-1][idx])
        return viterbi_seq

    def viterbi(self, ex, fwd_obs_logps, init_pmt_logps, pmt_logps, states, trans_matrix, ext_matrix):
        # Fake a init node
        trans_logps = [[0]]
        obs_logps = [[0]]
        step_states = [[]]
        seqlen = len(ex) + 1

        for t in range(len(ex)):
            # For each time step in permutated states transition
            b_idx = ex[t][0]
            e_idx = ex[t][1]
            t_logps = []; o_lops = []; s_states = []
            for i in range(b_idx, e_idx):
                if t == 0:
                    t_logps.append(init_pmt_logps[i])
                else:
                    t_logps.append(pmt_logps[i])
                o_lops.append(fwd_obs_logps[i])
                s_states.append(states[i])
            trans_logps.append(t_logps)
            obs_logps.append(o_lops)
            step_states.append(s_states)

        gama = [None]*(seqlen)
        arg_gama = [None]*(seqlen)
        gama[0] = [0]
        for t in range(1, seqlen):
            z_t = trans_logps[t]
            z_pre = trans_logps[t-1]
            o_t = obs_logps[t]
            s_t = step_states[t]
            s_pre = step_states[t-1]
            gama_pre = gama[t-1]
            gama[t] = []; arg_gama[t] = []
            for j in range(len(z_t)):
                scores_j = []
                for i in range(len(z_pre)):
                    if t > 1:
                        s = s_t[j][0]
                        pre_s = s_pre[i][-1]
                        inter_transition = trans_matrix[pre_s][s]
                        exter_transition = ext_matrix[pre_s][s]
                        #if self.fake_global and len(z_next) > 1 and len(z_t) > 1 and len(set(s_t[i])&set(s_next[j])) > 0:
                        #    continue
                        scores_j.append(gama_pre[i] + o_t[j] + (z_t[j] + z_pre[i] + inter_transition + exter_transition))
                    else:
                        s = s_t[j][0]
                        pre_s = self.hsmm_sid
                        exter_transition = ext_matrix[pre_s][s]
                        scores_j.append(gama_pre[i] + o_t[j] + (z_t[j] + z_pre[i]) + exter_transition)
                gama[t].append(max(scores_j))
                arg_gama[t].append(scores_j.index(max(scores_j)))
        return self.recover_bps(gama, arg_gama, step_states)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _obs_logprobs(self, output, target, tgt_idx, mask_tgt):
        bottled_output = self._bottle(output)
        scores = self.model.generator(bottled_output)
        ph_size, tgt_len = target.size()
        gtruth = target.contiguous().view(-1, 1)
        psk = scores.gather(1, gtruth).view(ph_size, tgt_len)
        lls_k = []
        for i in range(ph_size):
            b_p = tgt_idx[i][0]
            e_p = tgt_idx[i][1]
            psk_i = psk[i][b_p:e_p+1]
            #lls_i = psk_i.sum(0).log()
            lls_i = psk_i.log().sum(0)
            lls_k.append(lls_i)
        return torch.stack(lls_k)

    def _permut_logps(self, init_logps, trans_logps, states):
        pmt_logps = []; init_pmt_logps = []; pre_s = -1
        for state in states:
            init_pmt_score = []
            pmt_score = []
            for i in range(len(state)):
                s = state[i]
                if i == 0:
                    init_pmt_score.append(init_logps[s])
                    #pmt_score.append(trans_logps[self.hsmm_emis_id][s])
                else:
                    init_pmt_score.append(trans_logps[pre_s][s])
                    pmt_score.append(trans_logps[pre_s][s])
                pre_s = s
            init_pmt_score = torch.stack(init_pmt_score)
            init_pmt_logps.append(init_pmt_score.sum(0))

            if len(pmt_score) > 0:
                pmt_score = torch.stack(pmt_score)
                pmt_logps.append(pmt_score.sum(0))
            else:
                pmt_logps.append(torch.tensor(0.0, device=self.device))

        return torch.stack(init_pmt_logps), torch.stack(pmt_logps)

    def run(self, batch):
        src = batch.src
        tgt = batch.tgt
        pmt_msk = batch.pmt_msk
        states = batch.states
        ex_idx = batch.ex_idx
        tgt_idx = batch.tgt_idx
        mask_src = batch.mask_src
        mask_tgt = batch.mask_tgt

        outputs, _ = self.model(src, tgt, mask_src, pmt_msk, ex_idx)
        init_logps, trans_logps = self.model.trans_logprobs()
        init_pmt_logps, pmt_logps = self._permut_logps(init_logps, trans_logps, states)
        ext_logps = self.model.external_logprobs()

        target = batch.tgt[:,1:]
        fwd_obs_logps = self._obs_logprobs(outputs, target, tgt_idx, mask_tgt)

        viterbi_ress = []
        for ex in ex_idx:
            # For each example in a batch
            viterbi_res = self.viterbi(ex, fwd_obs_logps,
                                      init_pmt_logps,
                                      pmt_logps, states,
                                      trans_logps, ext_logps)
            viterbi_ress.append(viterbi_res)
        return viterbi_ress


class Annotator(object):

    def __init__(self, args, model, logger=None, dump_beam=""):
        self.logger = logger
        self.args = args
        self.model = model
        self.device = "cpu" if args.visible_gpus == '-1' else "cuda"

        self.start_token = self.model.tgt_dict['[unused0]']
        self.end_token = self.model.tgt_dict['[unused1]']

        self.tgt_id_to_tok = {}
        for tok in self.model.tgt_dict:
            tid = self.model.tgt_dict[tok]
            self.tgt_id_to_tok[tid] = tok

        self.src_id_to_tok = {}
        for tok in self.model.src_dict:
            tid = self.model.src_dict[tok]
            self.src_id_to_tok[tid] = tok

        self.id_to_rel = {}
        for tok in self.model.relation_dict:
            tid = self.model.relation_dict[tok]
            self.id_to_rel[tid] = tok

        self.viterbi_obj = Vierbi(self.model, self.device)


    def read_res(self, anno_results, batch):

        annotations = []
        batch_size = batch.batch_size
        for b in range(batch_size):
            anno_result = anno_results[b]
            anno_list = []
            for ann in anno_result:
                ann_str = '&&'.join([self.id_to_rel[int(a)] for a in ann])
                anno_list.append(ann_str)
            anno_str = '|'.join(anno_list)

            src = batch.src[b]
            raw_src = ' '.join([self.src_id_to_tok[int(t)] for t in src])

            ex_idx = batch.ex_idx[b]
            tgt = batch.tgt[ex_idx[-1][-1]-1]
            tgt = ' '.join([self.tgt_id_to_tok[int(n)] for n in tgt])

            annotation = (anno_str, raw_src, tgt)
            annotations.append(annotation)

        return annotations


    def annotate_batch(self, batch):
        with torch.no_grad():
            results = self.viterbi_obj.run(batch)
            return results


    def annotate(self, data_iter, step, attn_debug=False):
        self.model.eval()
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')
        anno_path = self.args.result_path + '.%d.anno' % step
        self.anno_out_file = codecs.open(anno_path, 'w', 'utf-8')

        with torch.no_grad():
            for batch in data_iter:
                batch_res = self.annotate_batch(batch)
                annotations = self.read_res(batch_res, batch)
                for trans in annotations:
                    anno, src, tgt = trans
                    src_str = src.replace('[unused0]', '')\
                                    .replace('[PAD]', '')\
                                    .replace('[unused1]', '')\
                                    .replace('[UNK]', '')
                    tgt_str = tgt.replace('[unused0]', '')\
                                    .replace('[PAD]', '')\
                                    .replace('[unused1]', '')\
                                    .replace('[UNK]', '')
                    src_str = re.sub(' +', ' ', src_str).strip()
                    tgt_str = re.sub(' +', ' ', tgt_str).strip()

                    self.src_out_file.write(src_str + '\n')
                    self.anno_out_file.write(tgt_str + '\t' + anno + '\n')

                self.src_out_file.flush()
                self.anno_out_file.flush()

        self.src_out_file.close()
        self.anno_out_file.close()
