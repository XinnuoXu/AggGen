#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import json
import torch

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer


def build_predictor(args, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')
    translator = Translator(args, model, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):

    def __init__(self, args, model, global_scorer=None, logger=None, dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator

        self.src_vocab = {}
        with open(args.src_dict_path) as f:
            line = f.read().strip()
            self.src_dict = json.loads(line)
        for key in self.src_dict:
            value = self.src_dict[key]
            self.src_vocab[value] = key

        self.vocab = {}
        with open(args.tgt_dict_path) as f:
            line = f.read().strip()
            self.tgt_dict = json.loads(line)
        for key in self.tgt_dict:
            value = self.tgt_dict[key]
            self.vocab[value] = key
        self.start_token = self.tgt_dict['[unused0]']
        self.end_token = self.tgt_dict['[unused1]']

        self.inference_mode = self.args.inference_mode
        self.nucleus_p = self.args.nucleus_p
        self.top_k = self.args.top_k

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        tensorboard_log_dir = args.model_path
        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}


    def from_batch(self, translation_batch, batch):
        preds, tgt_str, src = translation_batch["predictions"], batch.tgt_str, batch.src
        raw_srcs = batch.src_str
        batch_size = batch.batch_size
        translations = []
        for b in range(batch_size):
            pred_sents = [self.vocab[int(n)] for n in preds[b][0]]
            pred_sents = ' '.join(pred_sents)
            gold_sent = ' '.join(tgt_str[b].split())
            #raw_src = [self.src_vocab[int(t)] for t in src[b]][:500]
            #raw_src = ' '.join(raw_src)
            raw_src = '[CLS] ' + ' [SEP] '.join(raw_srcs[b]) + ' [SEP]'
            translation = (pred_sents, gold_sent, raw_src)
            translations.append(translation)
        return translations


    def translate(self, data_iter, step, attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data, batch)
                for trans in translations:
                    pred, gold, src = trans
                    pred_str = pred.replace('[unused0]', '')\
                                .replace('[unused1]', '')\
                                .replace('[PAD]', '')\
                                .replace('[SEP]', '')\
                                .replace('[UNK]', '')\
                                .replace(r' +', ' ').strip()
                    gold_str = gold.strip()
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)


    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict


    def translate_batch(self, batch, fast=False):
        with torch.no_grad():
            if self.inference_mode == 'beam':
                return self.beam_search(batch, self.max_length, self.min_length)
            elif self.inference_mode == 'nucleus':
                return self.sampling_decoder(batch, self.max_length, self.min_length, 'nucleus')
            elif self.inference_mode == 'topk':
                return self.sampling_decoder(batch, self.max_length, self.min_length, 'topk')


    def nucleus_sampling(self, samp_probs):
        ex_samp_probs = torch.exp(samp_probs)
        sorted_probs, sorted_indices = torch.sort(ex_samp_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.nucleus_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        sorted_samp_probs = sorted_probs.clone()
        sorted_samp_probs[sorted_indices_to_remove] = 0
        sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
        next_tokens = sorted_indices.gather(1, sorted_next_indices)
        next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()
        return next_tokens, next_logprobs


    def topk_sampling(self, samp_probs):
        samp_probs = torch.exp(samp_probs)
        indices_to_remove = samp_probs < torch.topk(samp_probs, self.top_k)[0][..., -1, None]
        samp_probs[indices_to_remove] = 0
        next_tokens = samp_probs.multinomial(1)
        next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
        return next_tokens, next_logprobs


    def sampling_decoder(self, batch, max_length, min_length, sample_method):

        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src

        src_features = self.model.encoder(src, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device
        alive_seq = torch.full([batch_size, 1], self.start_token, dtype=torch.long, device=device)
        ids = torch.tensor([i for i in range(batch_size)], device=device)

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)
            dec_out, dec_states, _ = self.model.decoder(decoder_input, src_features, dec_states, step=step)

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
            vocab_size = log_probs.size(-1)
            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            if sample_method == "nucleus":
                next_tokens, _ = self.nucleus_sampling(log_probs)
            elif sample_method == "topk":
                next_tokens, _ = self.topk_sampling(log_probs)
            alive_seq = torch.cat([alive_seq, next_tokens], dim=1)

            is_finished = next_tokens.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)

            if is_finished.any():
                is_finished = is_finished.view(-1)
                finished_hyp = is_finished.nonzero().view(-1)
                for b in finished_hyp:
                    results["predictions"][ids[b]].append(alive_seq[b])
                non_finished = is_finished.eq(0).nonzero().view(-1)
                if len(non_finished) == 0:
                    break
                alive_seq = alive_seq.index_select(0, non_finished).view(-1, alive_seq.size(-1))
                ids = ids.index_select(0, non_finished)
                src_features = src_features.index_select(0, non_finished)
                dec_states.map_batch_fn(lambda state, dim: state.index_select(dim, non_finished))

        return results



    def beam_search(self, batch, max_length, min_length=0):

        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src

        src_features = self.model.encoder(src, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(0, batch_size * beam_size,
                                    step=beam_size,
                                    dtype=torch.long,
                                    device=device)
        alive_seq = torch.full([batch_size * beam_size, 1],
                                self.start_token,
                                dtype=torch.long,
                                device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                                device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)
            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states, step=step)
            
            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if(self.args.block_trigram):
                cur_len = alive_seq.size(1)
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab[w] for w in words]
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results


class Translation(object):

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
