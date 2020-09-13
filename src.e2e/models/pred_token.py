#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import torch
import torch.nn as nn
from others.utils import tile


class Token_predictor(object):

    def __init__(self, args, model, global_scorer=None):
        self.args = args
        self.model = model
        self.generator = self.model.generator

        self.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        self.beam_size = args.beam_size
        self.start_token = self.model.tgt_dict['[unused0]']
        self.end_token = self.model.tgt_dict['[unused1]']
        self.pad_token = self.model.tgt_dict['[PAD]']
        self.sep_token = self.model.src_dict['[SEP]']
        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.tp_bias = args.tp_bias

        self.inference_mode = self.args.inference_mode
        self.nucleus_p = self.args.nucleus_p
        self.top_k = self.args.top_k

        self.tgt_id_to_tok = {}
        for tok in self.model.tgt_dict:
            tid = self.model.tgt_dict[tok]
            self.tgt_id_to_tok[tid] = tok

        self.softmax = nn.Softmax(dim=0)
        self.lm_cov_bias = self.args.lm_cov_bias


    def tokens_predictor(self, batch, batch_states):

        # Process Src
        srcs = batch.src
        mask_src = batch.mask_src
        batch_size = batch.batch_size
        src_features = self.model.encoder(srcs, mask_src)

        # Predict Tgt
        results = []
        for i in range(batch_size):
            src = srcs[i]
            src_feat = src_features[i]
            src_masks, state_seqs, binary_patts, patt_scores = batch_states[i]
            preds = []; scores = []
            for j in range(len(src_masks)):
                #print (state_seqs[j], binary_patts[j], patt_scores[j])
                # For each pattern candidates
                if self.inference_mode == 'beam':
                    tok_score, cov_score, pred = self._hmm_decoder(src, src_feat, src_masks[j])
                elif self.inference_mode == 'nucleus':
                    tok_score, pred = self._hmm_sampling(src, src_feat, src_masks[j], 'nucleus')
                elif self.inference_mode == 'topk':
                    tok_score, pred = self._hmm_sampling(src, src_feat, src_masks[j], 'topk')
                else:
                    return None
                #print (binary_patts[j], tok_score, cov_score)
                pat_score = patt_scores[j]
                score = pat_score * self.tp_bias + tok_score
                preds.append(pred)
                scores.append(score)
            sort_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            best_idx = sort_idx[0]
            result = {}
            result['pred'] = preds[best_idx]
            result['s_seq'] = state_seqs[best_idx]
            result['emission'] = binary_patts[best_idx]
            result['score'] = scores[best_idx]
            results.append(result)

        return results

    def _coverage_eva(self, src, src_mask, hmm_states, dec_seq):
        tgt_emb = self.model.decoder.embeddings(dec_seq)
        src_emb = self.model.encoder.embeddings(src[0])
        scores = torch.matmul(src_emb, tgt_emb.transpose(0, 1))
        mask = torch.cat([src_mask[sid].unsqueeze(1) for sid in hmm_states], dim=1)
        scores = scores.masked_fill(mask, -1e18)
        attn = self.softmax(scores)
        return attn.transpose(0, 1)

    def _expect_length(self, src, src_mask, hmm_states):
        beam_size = len(hmm_states)
        sep_toks = src.eq(self.sep_token)
        sep_toks = tile(sep_toks, beam_size, dim=0)
        pmt_mask = torch.cat([src_mask[sid].unsqueeze(0) for sid in hmm_states], dim=0)
        pmt_mask = ~pmt_mask
        tripple_nums = sep_toks & pmt_mask
        return tripple_nums.sum(dim=1)

    def _hmm_decoder(self, src, src_feat, src_masks):
        beam_size = self.beam_size
        device = self.device
        src = src.unsqueeze(0)
        src_len = src.size(1)
        src_feat = src_feat.unsqueeze(0)
        dec_states = self.model.decoder.init_decoder_state(src, src_feat, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(lambda state, dim: tile(state, beam_size, dim=dim))
        src_feat = tile(src_feat, beam_size, dim=0)
        beam_offset = torch.arange(0,
                            beam_size,
                            step=beam_size,
                            dtype=torch.long,
                            device=device)
        alive_seq = torch.full([beam_size, 1],
                            self.start_token,
                            dtype=torch.long,
                            device=device)
        coverage = torch.full([beam_size, src_len, 1],
                            0.0,
                            dtype=torch.float,
                            device=device)
        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1), device=device)
        # Structure that holds finished hypotheses.
        hmm_states = torch.tensor([0] * beam_size, device=device)
        phrase_len = torch.tensor([0] * beam_size, device=device)
        hmm_step_num = len(src_masks)
        hypotheses = []

        for step in range(self.max_length):
            src_mask = torch.cat([src_masks[sid].unsqueeze(0) for sid in hmm_states], dim=0)
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)
            dec_out, dec_states = self.model.decoder(
                                            decoder_input,
                                            src_feat,
                                            dec_states,
                                            step=step,
                                            memory_masks=src_mask)
            
            # Generator forward.
            probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
            log_probs = torch.log(probs)
            vocab_size = log_probs.size(-1)

            exp_lens = self._expect_length(src, src_masks, hmm_states)
            for i in range(len(phrase_len)):
                if phrase_len[i] < exp_lens[i] * self.min_length:
                    log_probs[i, self.end_token] = -10e20
                log_probs[i, self.pad_token] = -10e20

            # Multiply probs by the beam probability.
            if step == 0 or self.lm_cov_bias == 0:
                log_probs += topk_log_probs.view(-1).unsqueeze(1)
            else:
                cov_log = (coverage.max(dim=-1)[0].sum(dim=-1)/src_len).log()
                log_probs += (topk_log_probs*self.lm_cov_bias+cov_log).view(-1).unsqueeze(1)

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
                        words = [self.tgt_id_to_tok[w] for w in words]
                        words = ' '.join(words).split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            select_indices = topk_beam_index

            # Append last prediction.
            alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)
            cov_step = self._coverage_eva(src, src_masks, hmm_states, alive_seq[:, -1])
            coverage = torch.cat([coverage.index_select(0, select_indices), cov_step.unsqueeze(2)], -1)
            hmm_states = hmm_states.index_select(0, select_indices)
            phrase_len = phrase_len.index_select(0, select_indices)
            exp_lens = exp_lens.index_select(0, select_indices)

            # If phrase is finished or tgt is finished
            is_ph_finished = topk_ids.eq(self.end_token)
            for i in range(is_ph_finished.size(0)):
                if is_ph_finished[i]:
                    hmm_states[i] += 1
                    phrase_len[i] = 0
                else:
                    phrase_len[i] += 1
            is_finished = hmm_states.eq(hmm_step_num)
            if step + 1 == self.max_length:
                is_finished.fill_(1)

            # End condition is top beam is finished.
            end_condition = is_finished[0].eq(1)

            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(beam_size, alive_seq.size(-1))
                finished_hyp = is_finished.nonzero().view(-1)
                for j in finished_hyp:
                    cov_score = (coverage.max(dim=-1)[0].sum(dim=-1)/src_len).log()[j]
                    hypotheses.append((topk_scores[j], cov_score, predictions[j, 1:]))
                    topk_log_probs[j] = -10e20
                    hmm_states[j] = 0
                if end_condition:
                    best_hyp = sorted(hypotheses, key=lambda x: x[0], reverse=True)
                    score, cov_score, pred = best_hyp[0]
                    res_score = score
                    res_cov = cov_score
                    res_pred = pred
                    break
            src_feat = src_feat.index_select(0, select_indices)
            dec_states.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))

        return res_score, res_cov, res_pred


    def _nucleus_sampling(self, samp_probs):
        sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
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


    def _topk_sampling(self, samp_probs):
        samp_probs = torch.exp(samp_probs)
        indices_to_remove = samp_probs < torch.topk(samp_probs, self.top_k)[0][..., -1, None]
        samp_probs[indices_to_remove] = 0
        next_tokens = samp_probs.multinomial(1)
        next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
        return next_tokens, next_logprobs


    def _hmm_sampling(self, src, src_feat, src_masks, sample_method):
        device = self.device
        src = src.unsqueeze(0)
        src_feat = src_feat.unsqueeze(0)
        dec_states = self.model.decoder.init_decoder_state(src, src_feat, with_cache=True)
        alive_seq = torch.full([1, 1], self.start_token, dtype=torch.long, device=device)
        logprob = 0.0

        # Structure that holds finished hypotheses.
        hmm_states = 0
        phrase_len = 0
        hmm_step_num = len(src_masks)
        hypotheses = []

        for step in range(self.max_length):
            src_mask = src_masks[hmm_states].unsqueeze(0)
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)
            dec_out, dec_states = self.model.decoder(
                                            decoder_input,
                                            src_feat,
                                            dec_states,
                                            step=step,
                                            memory_masks=src_mask)
            
            # Generator forward.
            probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
            if phrase_len < self.min_length:
                probs[0, self.end_token] = 0
            probs[0, self.pad_token] = 0

            if sample_method == "nucleus":
                next_tokens, lprob = self._nucleus_sampling(probs)
            elif sample_method == "topk":
                next_tokens, lprob = self._topk_sampling(probs)
            alive_seq = torch.cat([alive_seq, next_tokens], dim=1)
            logprob += lprob

            # If phrase is finished
            is_ph_finished = next_tokens[0][0].eq(self.end_token)
            if is_ph_finished:
                hmm_states += 1
                phrase_len = 0
            else:
                phrase_len += 1
                
            # If tgt is finished
            is_finished = (hmm_states == hmm_step_num)
            if step + 1 == self.max_length:
                is_finished = 1

            # Save finished hypotheses.
            if is_finished:
                res_pred = alive_seq[0, 1:]
                res_score = logprob
                break

        return res_score, res_pred


