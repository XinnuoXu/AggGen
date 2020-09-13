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
from translate.beam import GNMTGlobalScorer
from models.pred_state import State_predictor
from models.pred_token import Token_predictor


def build_predictor(args, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')
    translator = Translator(args, model, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):

    def __init__(self, args, model, global_scorer=None, logger=None, dump_beam=""):
        self.logger = logger
        self.args = args
        self.model = model

        self.S_predictor = State_predictor(args, model)
        self.T_predictor = Token_predictor(args, model, global_scorer)

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


    def read_res(self, translation_results, batch):

        translations = []
        batch_size = batch.batch_size
        for b in range(batch_size):
            translation_result = translation_results[b]
            pred, score, state_seq, emission, tgt_str, src = \
                                    translation_result["pred"],\
                                    translation_result["score"],\
                                    translation_result["s_seq"],\
                                    translation_result["emission"],\
                                    batch.tgt_str[b], batch.src[b]
            # hmm info
            state_seq = ' '.join([self.id_to_rel[int(s)] for s in state_seq])
            emission = ' '.join([str(e) for e in emission])
            seq_info = state_seq + '\t' + emission
            # prediction
            pred_sents = ' '.join([self.tgt_id_to_tok[int(n)] for n in pred])
            # ground truth
            gold_sent = ' <ref_sep> '.join(tgt_str)
            # raw src
            raw_src = ' '.join([self.src_id_to_tok[int(t)] for t in src])
            translation = (pred_sents, seq_info, gold_sent, raw_src)
            translations.append(translation)

        return translations


    def translate_batch(self, batch):
        with torch.no_grad():
            batch_states = self.S_predictor.state_predictor(batch)
            results = self.T_predictor.tokens_predictor(batch, batch_states)
            return results


    def translate(self, data_iter, step, attn_debug=False):
        self.model.eval()
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')
        gold_path = self.args.result_path + '.%d.gold' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        can_path = self.args.result_path + '.%d.candidate' % step
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        hmm_path = self.args.result_path + '.%d.hmm' % step
        self.hmm_out_file = codecs.open(hmm_path, 'w', 'utf-8')

        with torch.no_grad():
            for batch in data_iter:
                batch_res = self.translate_batch(batch)
                translations = self.read_res(batch_res, batch)
                for trans in translations:
                    pred, hmm, gold, src = trans
                    pred_str = pred.replace('[unused0]', '')\
                                    .replace('[PAD]', '')\
                                    .replace('[unused1]', '')\
                                    .replace('[UNK]', '')
                    pred_str = re.sub(' +', ' ', pred_str).strip()
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold.strip() + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    self.hmm_out_file.write(hmm.strip() + '\n')
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                self.hmm_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()
        self.hmm_out_file.close()


    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

